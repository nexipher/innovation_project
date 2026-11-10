"""Run the 200-input g8 dev evaluation with Qwen + PROBE + TruFor.

The runner is deliberately separate from the 24-session acceptance runner:
it records full protocol identity before the first session, writes one
record per sample, and can resume without rerunning completed samples.

Usage:
  python scripts/g8_dev_eval.py --dry-run
  FORENSIC_ALLOW_GPU=1 python scripts/g8_dev_eval.py
  FORENSIC_ALLOW_GPU=1 python scripts/g8_dev_eval.py --resume
  FORENSIC_ALLOW_GPU=1 python scripts/g8_dev_eval.py --resume --retry-failed

No mock path exists here: the runner builds a real Qwen pipeline and refuses
anything whose client mode is not `qwen3_real`.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

DEFAULT_MANIFEST = ROOT / "sft_data/g8_dev_eval_20261110/manifest.json"
DEFAULT_OUT = ROOT / "sft_data/g8_dev_eval_20261110"
IGNORED_DIRTY_PREFIXES = ("sft_data/", "traces/")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha(payload) -> str:
    return sha256_text(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":"), default=str))[:16]


def load_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "dev_eval_manifest/1":
        raise SystemExit(f"unexpected manifest schema: {payload.get('schema')!r}")
    return payload


def verify_manifest(manifest: dict, manifest_path: Path) -> dict:
    samples = manifest.get("samples") or []
    if len(samples) != 200:
        raise SystemExit(f"manifest must contain 200 samples, got {len(samples)}")
    required = {"sample_id", "variant_id", "source_id", "family", "group",
                "path", "source_path", "q95_sha256", "source_sha256",
                "ground_truth", "task_truth"}
    ids, variants, paths, q95s = set(), set(), set(), set()
    failures = []
    for row in samples:
        missing = required - set(row)
        if missing:
            failures.append(f"{row.get('sample_id')}: missing fields {sorted(missing)}")
            continue
        for key, seen in (("sample_id", ids), ("variant_id", variants), ("path", paths)):
            value = row[key]
            if value in seen:
                failures.append(f"duplicate {key}: {value}")
            seen.add(value)
        q95 = sha256_file(ROOT / row["path"])
        if q95 != row["q95_sha256"]:
            failures.append(f"{row['sample_id']}: q95 hash mismatch")
        q95s.add(q95)
        source = sha256_file(ROOT / row["source_path"])
        if source != row["source_sha256"]:
            failures.append(f"{row['sample_id']}: source hash mismatch")
        if row.get("mask_path"):
            mask = sha256_file(ROOT / row["mask_path"])
            if mask != row.get("mask_sha256"):
                failures.append(f"{row['sample_id']}: mask hash mismatch")
    auto_sources = [row["source_id"] for row in samples
                    if row.get("family") == "AutoSplice_q95"]
    if len(auto_sources) != len(set(auto_sources)):
        failures.append("AutoSplice source repeated across real/forged")
    if failures:
        raise SystemExit("manifest verification failed:\n  " + "\n  ".join(failures[:20]))
    return {"samples": len(samples), "q95_hashes": len(q95s),
            "masks": sum(1 for row in samples if row.get("mask_path")),
            "manifest_sha256": sha256_file(manifest_path)}


def git_state(out: Path) -> dict:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            capture_output=True, text=True).stdout.splitlines()
    out_rel = os.path.relpath(out, ROOT)
    dirty = []
    for line in status:
        path = line[3:].strip() if len(line) > 3 else line.strip()
        if path.startswith(out_rel + "/") or path == out_rel:
            continue
        dirty.append(line)
    return {"commit": commit, "dirty": bool(dirty), "dirty_paths": dirty}


def cgroup_state() -> dict:
    result = {}
    for name in ("memory.max", "memory.high", "memory.current"):
        path = Path("/sys/fs/cgroup") / name
        try:
            result[name] = path.read_text().strip()
        except OSError:
            result[name] = None
    return result


def build_runtime_identity(fsm) -> dict:
    from utils.protocol_identity import (
        identity_sha256, runtime_identity)

    identity = runtime_identity(
        fsm, tools_policy="probe+trufor", greedy=True,
        inference_mode="non-thinking")
    prompt = getattr(fsm._mllm, "system_prompt", "") or ""
    actual = {
        "sha256": sha256_text(prompt),
        "digest": sha256_text(prompt)[:16],
        "chars": len(prompt),
    }
    identity["prompts"]["actual"] = actual
    # The manifest path/hash is injected by the caller; this function only
    # owns the runtime/model/expert identity.
    return {"identity": identity, "identity_sha256": identity_sha256(identity),
            "actual_prompt": actual}


def build_batch_meta(fsm, manifest_path: Path, out: Path, batch_id: str,
                     manifest: dict, verification: dict) -> dict:
    from utils.protocol_identity import expert_identity

    runtime = build_runtime_identity(fsm)
    experts = expert_identity(getattr(fsm, "_experts", {}) or {})
    # TruFor is registered in the runtime identity but its preprocessing text is
    # not in score_semantics; record the adapter's actual input recipe here.
    if "trufor_expert" in experts:
        experts["trufor_expert"]["preprocess_detail"] = (
            "RGB source image saved as PNG; TruFor test.py forwards it directly "
            "(model resize/normalize inside); outputs score/map/conf")
        experts["trufor_expert"]["preprocess_sha256"] = sha256_text(
            experts["trufor_expert"]["preprocess_detail"])
    identity = runtime["identity"]
    identity["experts"] = experts
    identity["manifest"] = {
        "path": os.path.relpath(manifest_path, ROOT),
        "sha256": verification["manifest_sha256"],
    }
    from utils.protocol_identity import identity_sha256
    identity_sha = identity_sha256(identity)
    return {
        "batch_id": batch_id,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": {
            "path": os.path.relpath(manifest_path, ROOT),
            "sha256": verification["manifest_sha256"],
            "samples": verification["samples"],
            "seed": manifest.get("seed"),
            "counts": manifest.get("counts"),
        },
        "code": git_state(out),
        "runtime_identity": identity,
        "model": identity.get("model"),
        "prompts": identity.get("prompts"),
        "identity_sha256": identity_sha,
        "config_sha256": identity_sha,
        "actual_prompt": runtime["actual_prompt"],
        "experts": experts,
        "cgroup": cgroup_state(),
        "tools": ["probe_dino_expert", "trufor_expert"],
        "decoding": {"greedy": True, "inference_mode": "non-thinking",
                     "do_sample": False, "temperature": 0.0,
                     "max_new_tokens": 1024},
        "expected_seconds_per_session": 56.0,
        "expected_total_seconds": round(56.0 * verification["samples"], 1),
        "resume_policy": "completed records are skipped; failed records are preserved",
    }


def record_page_name(index: int, sample_id: str) -> str:
    return f"{index:03d}__{sample_id}.html"


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def make_record(sample: dict, index: int, result: dict, trace: dict,
                elapsed: float, config_sha: str, attempt: int) -> dict:
    verdict = result.get("final_verdict") or {}
    trace_meta = trace.get("metadata") or {}
    record = {
        "index": index,
        "sample_id": sample["sample_id"],
        "variant_id": sample.get("variant_id"),
        "source_id": sample.get("source_id"),
        "family": sample.get("family"),
        "group": sample.get("group"),
        "source_role": sample.get("source_role"),
        "split": sample.get("split"),
        "old_split": sample.get("old_split"),
        "no_known_use": sample.get("no_known_use"),
        "usage_flags": sample.get("usage_flags") or [],
        "usage_roles": sample.get("usage_roles") or [],
        "condition": sample.get("condition"),
        "treatment": "jpeg_q95",
        "bytes": sample.get("bytes"),
        "width": sample.get("width"),
        "height": sample.get("height"),
        "encoder": sample.get("encoder"),
        "selection_role": sample.get("selection_role"),
        "authentic_from_forged_base": sample.get("authentic_from_forged_base"),
        "q95_sha256": sample.get("q95_sha256"),
        "source_sha256": sample.get("source_sha256"),
        "mask_path": sample.get("mask_path"),
        "mask_sha256": sample.get("mask_sha256"),
        "mask_shared_by_base": sample.get("mask_shared_by_base"),
        "base_variant_count": sample.get("base_variant_count"),
        "variant": sample.get("variant"),
        "ground_truth": sample.get("ground_truth"),
        "task_truth": sample.get("task_truth"),
        "risk_tags": sample.get("risk_tags") or [],
        "config_sha256": config_sha,
        "attempt": attempt,
        "status": "completed",
        "trajectory_id": result.get("session_id"),
        "final_verdict": verdict.get("verdict"),
        "verdict_status": verdict.get("verdict_status"),
        "report": verdict.get("report"),
        "primary_evidence": verdict.get("primary_evidence"),
        "whole_image_generation": verdict.get("whole_image_generation"),
        "local_manipulation": verdict.get("local_manipulation"),
        "suspicious_regions": verdict.get("suspicious_regions"),
        "suspicious_region_review_flags":
            verdict.get("suspicious_region_review_flags"),
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
            "untried_experts": trace_meta.get("untried_experts") or [],
            "failed_experts": trace_meta.get("failed_experts") or [],
            "preclose_checks": trace_meta.get("preclose_checks") or [],
            "abstention_cited_untried_expert":
                trace_meta.get("abstention_cited_untried_expert") or [],
        },
        "elapsed_s": round(elapsed, 2),
        "image_path": sample.get("path"),
        "source_image": sample.get("source_path"),
        "trace_path": os.path.relpath(Path(result["sft_data_path"]), DEFAULT_OUT),
        "evidence": result.get("evidence_chain") or [],
        "evidence_check": verdict.get("evidence_check"),
        "coverage_check": verdict.get("coverage_check"),
        "certainty_check": verdict.get("certainty_check"),
        "tools_served": ["probe_dino_expert", "trufor_expert"],
    }
    if "certainty" in verdict:
        record["certainty"] = verdict.get("certainty")
        record["verdict_schema"] = verdict.get("verdict_schema")
    if "confidence" in verdict:
        record["confidence"] = verdict.get("confidence")
    return record


def build_index(out: Path, batch_id: str, records: list[dict],
                batch_meta: dict) -> None:
    links = []
    for row in records:
        page = f"{row['index']:03d}__{row['sample_id']}.html"
        links.append(
            f'<li><a href="{html.escape(page)}">{html.escape(row["sample_id"])}</a> '
            f'— {html.escape(str(row.get("ground_truth")))}; '
            f'whole={html.escape(str(row.get("whole_image_generation")))}; '
            f'local={html.escape(str(row.get("local_manipulation")))}; '
            f'certainty={html.escape(str(row.get("certainty")))}; '
            f'calls={row.get("counters", {}).get("expert_calls")}; '
            f'stop={html.escape(str(row.get("halting_reason")))}</li>')
    index = f"""<!doctype html><html><head><meta charset="utf-8">
<title>g8 dev evaluation — {html.escape(batch_id)}</title></head><body>
<h1>g8 200-source development evaluation（{html.escape(batch_id)}）</h1>
<p><b>development evaluation</b>；不是人工验收，不是 SFT 数据。逐条查看原始对话、
实际投送图片、三图同位置观察、预测框叠加、专家结果、纠正过程与停止原因。</p>
<p>model: {html.escape(json.dumps(batch_meta.get("runtime_identity", {}).get("model", {}), ensure_ascii=False))}<br>
prompt actual sha256: {html.escape(batch_meta.get("actual_prompt", {}).get("sha256", ""))}<br>
prompt digest: {html.escape(batch_meta.get("runtime_identity", {}).get("prompts", {}).get("digest", ""))}<br>
config: {html.escape(batch_meta.get("config_sha256", ""))}</p>
<ul>{''.join(links)}</ul>
<p><a href="records.json" download>下载 records.json</a> ·
<a href="review_records.csv" download>下载 review_records.csv</a></p>
</body></html>"""
    (out / "index.html").write_text(index, encoding="utf-8")


def write_csv(out: Path, records: list[dict]) -> None:
    import csv
    fields = ["index", "sample_id", "variant_id", "family", "group",
              "source_role", "ground_truth", "whole_image_generation",
              "local_manipulation", "final_verdict", "certainty",
              "verdict_status", "halting_reason", "mask_path",
              "suspicious_regions", "joint_verdict_issues",
              "coverage_check", "certainty_check", "counters", "attempt"]
    with (out / "review_records.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in records:
            writer.writerow({key: json.dumps(row.get(key), ensure_ascii=False)
                             if isinstance(row.get(key), (dict, list)) else row.get(key)
                             for key in fields})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--batch-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    out = Path(args.out).resolve()
    manifest = load_manifest(manifest_path)
    verification = verify_manifest(manifest, manifest_path)
    samples = manifest["samples"]
    if args.limit:
        samples = samples[:args.limit]
    if args.dry_run:
        print(json.dumps({
            "manifest": str(manifest_path),
            "samples": len(samples),
            "counts": manifest.get("counts"),
            "verification": verification,
            "estimated_seconds": round(56.0 * len(samples), 1),
            "output": str(out),
        }, ensure_ascii=False, indent=2))
        return 0

    if os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1 后再运行")
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "PROBE_WEIGHTS",
        "/root/autodl-tmp/third_party/probe/DINOv2_best_model_step_34999.pth")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    os.environ.setdefault("TRUFOR_RESULTS", str(out / "trufor_raw"))
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    batch_id = args.batch_id or out.name
    records_dir = out / "records"
    failures_dir = out / "failures"
    records_dir.mkdir(exist_ok=True)
    failures_dir.mkdir(exist_ok=True)
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from main import build_pipeline
    from utils.logger import SessionLogger
    from utils.review_store import sample_fingerprint
    from scripts.g8_review_workstation import session_page

    fsm = build_pipeline("qwen", greedy=True, inference_mode="non-thinking",
                         local_files_only=True, tools=["probe", "trufor"])
    mllm = getattr(fsm, "_mllm", None)
    if getattr(mllm, "mode", None) != "qwen3_real":
        raise SystemExit("refusing a non-real MLLM session (mock or unknown mode)")
    batch_meta = build_batch_meta(fsm, manifest_path, out, batch_id, manifest, verification)
    batch_meta["worktree_allowed_dirty"] = args.allow_dirty
    meta_path = out / "batch_meta.json"
    identity_path = out / "config_identity.json"
    if meta_path.exists():
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise SystemExit(f"{meta_path} exists; use --resume or a new --out")
        if previous.get("manifest", {}).get("sha256") != verification["manifest_sha256"]:
            raise SystemExit("resume refused: manifest changed since batch start")
        if previous.get("config_sha256") != batch_meta["config_sha256"]:
            raise SystemExit("resume refused: runtime/config identity changed since batch start")
        if previous.get("code", {}).get("commit") != batch_meta["code"].get("commit"):
            raise SystemExit("resume refused: code commit changed since batch start")
        if batch_meta["code"]["dirty"] and not args.allow_dirty:
            raise SystemExit(
                "resume refused: non-output worktree changed:\n  "
                + "\n  ".join(batch_meta["code"]["dirty_paths"]))
        batch_meta = previous
        print(f"resume: {meta_path} identity verified", flush=True)
    else:
        if batch_meta["code"]["dirty"] and not args.allow_dirty:
            raise SystemExit(
                "worktree has non-output changes; commit them before the batch:\n  "
                + "\n  ".join(batch_meta["code"]["dirty_paths"]))
        write_json_atomic(meta_path, batch_meta)
        write_json_atomic(identity_path, batch_meta["runtime_identity"])
        print(f"wrote {meta_path}", flush=True)

    completed = {}
    for path in sorted(records_dir.glob("*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if row.get("status") == "completed":
            completed[row["sample_id"]] = row

    failed_ids = set()
    failures_path = out / "failures.jsonl"
    if failures_path.exists():
        for line in failures_path.read_text(encoding="utf-8").splitlines():
            try:
                failed_ids.add(json.loads(line)["sample_id"])
            except (ValueError, KeyError):
                continue

    print(json.dumps({
        "batch_id": batch_id, "samples": len(samples),
        "completed": len(completed), "failed_recorded": len(failed_ids),
        "config_sha256": batch_meta["config_sha256"],
        "estimated_seconds": round(56.0 * (len(samples) - len(completed)), 1),
        "out": str(out),
    }, ensure_ascii=False, indent=2), flush=True)

    started = time.time()
    for index, sample in enumerate(samples, 1):
        sid = sample["sample_id"]
        if sid in completed:
            continue
        if sid in failed_ids and not args.retry_failed:
            print(f"[{index}/{len(samples)}] {sid} skipped: previously failed", flush=True)
            continue
        attempt = 1
        while (failures_dir / f"{index:03d}__{sid}__attempt{attempt}.json").exists():
            attempt += 1
        logger = SessionLogger(sft_dir=str(out / "traces"))
        fsm._logger = logger
        t0 = time.perf_counter()
        try:
            result = fsm.run(str(ROOT / sample["path"]),
                             ground_truth=sample["ground_truth"])
            elapsed = time.perf_counter() - t0
            trace_path = Path(result["sft_data_path"])
            trace = json.loads(trace_path.read_text(encoding="utf-8")) \
                if trace_path.exists() else {}
            record = make_record(sample, index, result, trace, elapsed,
                                 batch_meta["config_sha256"], attempt)
            record["trace_path"] = os.path.relpath(trace_path, out)
            try:
                from utils.review_overlays import render_record_overlays
                record["review_overlays"] = render_record_overlays(
                    record, str(out / "review_overlays"))
            except Exception as exc:
                record["review_overlays"] = []
                record["review_overlay_error"] = f"{type(exc).__name__}: {exc}"
            page_name = record_page_name(index, sid)
            sample_for_page = {
                "record": record,
                "trace": trace,
                "page": page_name,
                "fingerprint": sample_fingerprint(record, trace),
                "source_image": sample["source_path"],
            }
            page = session_page(
                sample_for_page, str(out), "index.html", batch_meta=batch_meta,
                meta={"schema": "dev_eval/1", "generation": "g8",
                      "batch": batch_id, "samples": {}, "repo_decisions": {}})
            (out / page_name).write_text(page, encoding="utf-8")
            record["page"] = page_name
            write_json_atomic(records_dir / f"{index:03d}__{sid}.json", record)
            completed[sid] = record
            print(f"[{index}/{len(samples)}] {sid} "
                  f"overall={record['final_verdict']} whole={record['whole_image_generation']} "
                  f"local={record['local_manipulation']} certainty={record.get('certainty')} "
                  f"calls={record['counters']['expert_calls']} {elapsed:.1f}s", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - t0
            failure = {
                "index": index,
                "sample_id": sid,
                "variant_id": sample.get("variant_id"),
                "source_id": sample.get("source_id"),
                "attempt": attempt,
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(elapsed, 2),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-8000:],
                "config_sha256": batch_meta["config_sha256"],
                "status": "failed",
            }
            write_json_atomic(
                failures_dir / f"{index:03d}__{sid}__attempt{attempt}.json", failure)
            with failures_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(failure, ensure_ascii=False) + "\n")
            failed_ids.add(sid)
            print(f"[{index}/{len(samples)}] {sid} FAILED: {failure['error']}", flush=True)

    ordered = [completed[s["sample_id"]] for s in samples if s["sample_id"] in completed]
    write_json_atomic(out / "records.json", ordered)
    (out / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered),
        encoding="utf-8")
    write_csv(out, ordered)
    build_index(out, batch_id, ordered, batch_meta)
    end = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "code": git_state(out),
        "config_sha256": batch_meta["config_sha256"],
        "completed": len(ordered), "expected": len(samples),
        "failed": len(failed_ids),
    }
    write_json_atomic(out / "run_end.json", end)
    (out / "REVIEW_STATUS.md").write_text(
        f"# g8 dev evaluation — {batch_id}\n\n"
        f"- sessions: {len(ordered)} / {len(samples)} completed\n"
        f"- generated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"- status: development evaluation; not human acceptance, not SFT data\n"
        f"- config: `{batch_meta['config_sha256']}`\n"
        f"- workbench: `index.html`\n"
        f"- records: `records.json`, `records.jsonl` (if present), `review_records.csv`\n"
        f"- failures: `failures.jsonl` / `failures/`\n"
        f"- summary: `SUMMARY.md` (generated by `scripts/analyze_dev_eval.py`)\n",
        encoding="utf-8")
    print(json.dumps({
        "out": str(out), "completed": len(ordered),
        "failed": len(failed_ids), "elapsed_s": round(time.time() - started, 1),
        "config_sha256": batch_meta["config_sha256"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(ordered) == len(samples) else 1


if __name__ == "__main__":
    raise SystemExit(main())
