"""Run the fixed 24-session PROBE+TruFor joint-task acceptance batch.

This produces traces and a review page.  It does **not** claim human acceptance:
the page says pending human review until a person dispositions it.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "sft_data/g8_acceptance/dual_20261107/manifest.json"
OUT = ROOT / "sft_data/g8_acceptance/dual_20261107"


def _rel(path: str, page_dir: Path) -> str:
    return os.path.relpath(str(path), str(page_dir))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(MANIFEST))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--batch-id", default=None,
                        help="Batch label written into pages/metadata (default: out basename).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manifest = json.load(open(args.manifest, encoding="utf-8"))
    samples = manifest["samples"]
    if args.limit:
        samples = samples[:args.limit]
    if args.dry_run:
        print(json.dumps({"samples": len(samples),
                          "ids": [row["sample_id"] for row in samples]},
                         ensure_ascii=False, indent=2))
        return 0

    if os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1（联合验收）")
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    batch_id = args.batch_id or out.name
    # Keep the exact input manifest beside the generated batch so a later
    # reader can reproduce the same 24 inputs without guessing.
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.environ.setdefault("PROBE_WEIGHTS",
                          "/root/autodl-tmp/third_party/probe/"
                          "DINOv2_best_model_step_34999.pth")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    os.environ.setdefault("TRUFOR_RESULTS", str(out / "trufor_raw"))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

    from main import build_pipeline
    from utils.logger import SessionLogger
    from utils.review_store import sample_fingerprint
    from scripts.g8_review_workstation import session_page

    fsm = build_pipeline("qwen", greedy=True, inference_mode="non-thinking",
                         local_files_only=True, tools=["probe", "trufor"])
    identity = {}
    if callable(getattr(fsm._mllm, "identity", None)):
        try:
            identity = fsm._mllm.identity()
        except Exception:
            identity = {}
    prompt = getattr(fsm._mllm, "system_prompt", "") or ""
    prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()[:16]
    batch_meta = {
        "model": identity,
        "experts": {name: {"class": type(expert).__name__,
                            "device": str(getattr(expert, "device", ""))}
                    for name, expert in fsm._experts.items()},
        "prompts": {"digest": prompt_digest,
                    "tools": list(fsm._experts)},
        "batch_id": batch_id,
        "manifest": str(Path(args.manifest).resolve()),
    }
    (out / "batch_meta.json").write_text(
        json.dumps(batch_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    records = []
    started = time.time()
    for index, sample in enumerate(samples, 1):
        logger = SessionLogger(sft_dir=str(out / "traces"))
        fsm._logger = logger
        t0 = time.perf_counter()
        result = fsm.run(str(ROOT / sample["path"]),
                         ground_truth=sample["ground_truth"])
        elapsed = round(time.perf_counter() - t0, 2)
        trace_path = Path(result["sft_data_path"])
        trace = json.load(open(trace_path, encoding="utf-8")) if trace_path.exists() else {}
        verdict = result.get("final_verdict") or {}
        record = {
            "trajectory_id": result["session_id"],
            "variant_id": sample["sample_id"],
            "source_id": sample["source_id"],
            "family": sample["family"],
            "split": "acceptance_dev",
            "generator": sample.get("group"),
            "treatment": "jpeg_q95",
            "ground_truth": sample["ground_truth"],
            "task_truth": sample["task_truth"],
            "risk_tags": sample.get("risk_tags") or [],
            "final_verdict": verdict.get("verdict"),
            "verdict_status": verdict.get("verdict_status"),
            "confidence": verdict.get("confidence"),
            "report": verdict.get("report"),
            "primary_evidence": verdict.get("primary_evidence"),
            "whole_image_generation": verdict.get("whole_image_generation"),
            "local_manipulation": verdict.get("local_manipulation"),
            "suspicious_regions": verdict.get("suspicious_regions"),
            "joint_verdict_issues": verdict.get("joint_verdict_issues"),
            "model_candidate": verdict.get("model_candidate"),
            "halting_reason": result.get("halting_reason"),
            "policy_reasons": verdict.get("policy_reasons"),
            "counters": {
                "model_turns": result.get("model_turn_count"),
                "expert_calls": result.get("expert_call_count"),
                "unique_evidence": result.get("unique_evidence_count"),
                "skipped_duplicate_calls": result.get("skipped_duplicate_calls"),
                "suppressed_duplicate_count": result.get("suppressed_duplicate_count"),
                "weighted_cost": result.get("weighted_cost"),
            },
            "elapsed_s": elapsed,
            "image_path": sample["path"],
            "source_image": sample["source_path"],
            "trace_path": _rel(trace_path, out),
            "evidence": result.get("evidence_chain") or [],
            "evidence_check": verdict.get("evidence_check"),
            "tools_served": list(fsm._experts),
        }
        page_name = f"{index:02d}__{sample['sample_id']}.html"
        record["trace_path"] = _rel(trace_path, out)
        try:
            from utils.review_overlays import render_record_overlays
            record["review_overlays"] = render_record_overlays(
                record, str(out / "review_overlays"))
        except Exception as exc:  # review aid must not abort the batch
            record["review_overlays"] = []
            record["review_overlay_error"] = f"{type(exc).__name__}: {exc}"
        sample_for_page = {
            "record": record,
            "trace": trace,
            "page": page_name,
            "fingerprint": sample_fingerprint(record, trace),
            "source_image": sample["source_path"],
        }
        page = session_page(
            sample_for_page, str(out), "index.html", batch_meta=batch_meta,
            meta={"schema": "dual_acceptance/1", "generation": "g8",
                  "batch": batch_id, "samples": {}, "repo_decisions": {}})
        (out / page_name).write_text(page, encoding="utf-8")
        records.append(record)
        print(f"[{index}/{len(samples)}] {sample['sample_id']} "
              f"verdict={record['final_verdict']} whole={record['whole_image_generation']} "
              f"local={record['local_manipulation']} calls={record['counters']['expert_calls']} "
              f"{elapsed}s")

    records_path = out / "records.json"
    records_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    links = "\n".join(
        f'<li><a href="{html.escape(f"{i+1:02d}__{row["variant_id"]}.html")}">'
        f'{html.escape(row["variant_id"])}</a> — '
        f'{html.escape(str(row["ground_truth"]))}; '
        f'whole={html.escape(str(row["whole_image_generation"]))}; '
        f'local={html.escape(str(row["local_manipulation"]))}; '
        f'calls={row["counters"]["expert_calls"]}; '
        f'stop={html.escape(str(row["halting_reason"]))}</li>'
        for i, row in enumerate(records))
    index = f"""<!doctype html><html><head><meta charset="utf-8">
<title>PROBE+TruFor joint acceptance — {html.escape(batch_id)}</title></head><body>
<h1>PROBE + TruFor 联合任务人工验收（{html.escape(batch_id)}；pending human review）</h1>
<p>固定 {len(records)} 个会话；禁止把自动检查当作人工验收。逐条打开核对：模型是否理解
PROBE/TruFor 的分别作用、是否合理追加工具、是否修正初始怀疑、解释是否忠实。</p>
<p>模型：{html.escape(json.dumps(identity, ensure_ascii=False))}<br>
prompt digest: {html.escape(prompt_digest)}；tools: probe, trufor</p>
<ul>{links}</ul>
<p><a href="records.json" download>下载 records.json</a></p>
</body></html>"""
    (out / "index.html").write_text(index, encoding="utf-8")
    (out / "REVIEW_STATUS.md").write_text(
        f"# PROBE + TruFor joint acceptance — {batch_id}\n\n"
        f"- sessions: {len(records)}\n"
        f"- generated: {datetime.now().isoformat(timespec='seconds')}\n"
        "- status: **pending human review**\n"
        "- workbench: `index.html`\n"
        "- records: `records.json`\n"
        "- This batch is acceptance material, not SFT data; do not train on it.\n",
        encoding="utf-8")
    print(json.dumps({"out": str(out), "sessions": len(records),
                      "elapsed_s": round(time.time() - started, 1),
                      "status": "pending_human_review"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
