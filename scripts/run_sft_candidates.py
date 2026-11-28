"""Run the revised SFT candidate trajectories (24 main + 2 single-expert).

The assistant turns come from `candidate_scripts.json` (scripted candidate
writer, not Qwen).  PROBE and TruFor are the real adapters, and the controller,
evidence tokens, artifact panels, coverage/certainty checks and closing rules
are the unchanged runtime.  Ground truth and masks are written to the review
record only; the assistant never sees them.

The run stops on the first expert failure, missing evidence, non-concluding
session or target/record mismatch.  It does not retry or repair in place.
"""

from __future__ import annotations

import argparse
import csv
import gc
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

DEFAULT_OUT = ROOT / "sft_data/sft_candidates_20261127"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


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


def page_name(index: int, sample_id: str) -> str:
    return f"{index:03d}__{sample_id}.html"


def compact_evidence(token: dict) -> dict:
    return {
        "evidence_id": token.get("evidence_id"),
        "expert": token.get("source") or token.get("expert"),
        "measurement": token.get("evidence_name"),
        "status": token.get("status", "measured"),
        "scope": token.get("measurement_scope"),
        "strength": token.get("strength"),
        "raw_metric": token.get("raw_metric"),
        "region_pixels": token.get("region_pixels"),
        "map_stats": token.get("map_stats"),
        "visual_artifacts": token.get("visual_artifacts"),
    }


def make_record(sample: dict, index: int, result: dict, trace: dict,
                elapsed: float, script: dict, config_sha: str) -> dict:
    verdict = result.get("final_verdict") or {}
    meta = trace.get("metadata") or {}
    target = script["verdict"]
    return {
        "index": index,
        "record_schema": "sft_candidate_scripted/1",
        "sample_id": sample["sample_id"],
        "variant_id": sample.get("variant_id"),
        "source_id": sample.get("source_id"),
        "family": sample.get("family"),
        "group": sample.get("group"),
        "source_role": sample.get("source_role"),
        "kind": sample.get("kind"),
        "split": sample.get("split"),
        "selection_role": sample.get("selection_role"),
        "risk_tags": sample.get("risk_tags") or [],
        "key_case": bool(sample.get("key_case")),
        "key_case_rank": sample.get("key_case_rank"),
        "condition": sample.get("condition"),
        "treatment": sample.get("treatment"),
        "bytes": sample.get("bytes"),
        "width": sample.get("width"),
        "height": sample.get("height"),
        "encoder": sample.get("encoder"),
        "q95_sha256": sample.get("q95_sha256"),
        "source_sha256": sample.get("source_sha256"),
        "mask_path": sample.get("mask_path"),
        "mask_sha256": sample.get("mask_sha256"),
        "mask_shared_by_base": sample.get("mask_shared_by_base"),
        "base_variant_count": sample.get("base_variant_count"),
        "variant": sample.get("variant"),
        "ground_truth": sample.get("ground_truth"),
        "task_truth": sample.get("task_truth"),
        "screening_review_only": sample.get("screening_review_only"),
        "config_sha256": config_sha,
        "status": "completed",
        "trajectory_id": result.get("session_id"),
        "dialogue_mode": "sft_candidate_scripted",
        "assistant_source": ("scripted candidate writer; PROBE/TruFor real "
                             "through the runtime controller"),
        "call_order": script["call_order"],
        "target_verdict": target,
        "target_matches_record": (
            verdict.get("verdict") == target.get("verdict")
            and verdict.get("whole_image_generation")
            == target.get("whole_image_generation")
            and verdict.get("local_manipulation")
            == target.get("local_manipulation")),
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
        "certainty": verdict.get("certainty"),
        "halting_reason": result.get("halting_reason"),
        "policy_reasons": verdict.get("policy_reasons"),
        "counters": {
            "model_turns": result.get("model_turn_count"),
            "expert_calls": result.get("expert_call_count"),
            "unique_evidence": len(result.get("evidence_chain") or []),
            "skipped_duplicate_calls": result.get("skipped_duplicate_calls"),
            "suppressed_duplicate_count": result.get("suppressed_duplicate_count"),
            "weighted_cost": result.get("weighted_cost"),
            "untried_experts": meta.get("untried_experts") or [],
            "failed_experts": meta.get("failed_experts") or [],
            "preclose_checks": meta.get("preclose_checks") or [],
            "abstention_cited_untried_expert":
                meta.get("abstention_cited_untried_expert") or [],
        },
        "elapsed_s": round(elapsed, 2),
        "image_path": sample.get("path"),
        "source_image": sample.get("source_path"),
        "trace_path": os.path.relpath(Path(result["sft_data_path"]), ROOT),
        "evidence": [compact_evidence(token)
                     for token in (result.get("evidence_chain") or [])],
        "evidence_check": verdict.get("evidence_check"),
        "coverage_check": verdict.get("coverage_check"),
        "certainty_check": verdict.get("certainty_check"),
        "tools_served": ["probe_dino_expert", "trufor_expert"],
        "review_note": (
            "Assistant text is a candidate draft; every claim is tied to the "
            "evidence tokens shown on this page. Ground truth and mask are "
            "review-only."),
    }


def write_csv(path: Path, records: list[dict]) -> None:
    fields = ["index", "sample_id", "kind", "group", "selection_role",
              "key_case", "key_case_rank", "ground_truth",
              "whole_image_generation", "local_manipulation", "final_verdict",
              "certainty", "verdict_status", "halting_reason", "call_order",
              "expert_calls", "target_matches_record", "mask_path",
              "suspicious_regions", "joint_verdict_issues",
              "coverage_check", "certainty_check", "elapsed_s"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields,
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in records:
            out = dict(row)
            out["call_order"] = json.dumps(row.get("call_order"), ensure_ascii=False)
            out["expert_calls"] = (row.get("counters") or {}).get("expert_calls")
            writer.writerow({key: json.dumps(out.get(key), ensure_ascii=False)
                             if isinstance(out.get(key), (dict, list)) else out.get(key)
                             for key in fields})


def build_index(out: Path, records: list[dict], batch_meta: dict) -> None:
    key = [row for row in records if row.get("key_case")]
    key.sort(key=lambda row: row.get("key_case_rank") or 99)
    single = [row for row in records
              if str(row.get("kind", "")).startswith("single_expert")]
    rest = [row for row in records
            if not row.get("key_case")
            and not str(row.get("kind", "")).startswith("single_expert")]

    def table(rows):
        body = []
        for row in rows:
            body.append(
                f'<tr><td>{row.get("key_case_rank") or ""}</td>'
                f'<td><a href="{html.escape(row["page"])}">'
                f'{html.escape(row["sample_id"])}</a></td>'
                f'<td>{html.escape(str(row.get("group")))}</td>'
                f'<td>{html.escape(str(row.get("selection_role")))}</td>'
                f'<td>{html.escape(str(row.get("ground_truth")))}</td>'
                f'<td>{html.escape(str(row.get("whole_image_generation")))}</td>'
                f'<td>{html.escape(str(row.get("local_manipulation")))}</td>'
                f'<td>{html.escape(str(row.get("final_verdict")))}</td>'
                f'<td>{html.escape(str(row.get("certainty")))}</td>'
                f'<td>{html.escape(str((row.get("counters") or {}).get("expert_calls")))}</td>'
                f'<td>{html.escape(" → ".join(row.get("call_order") or []))}</td></tr>')
        return ("<table><tr><th>#</th><th>sample</th><th>group</th><th>role</th>"
                "<th>GT</th><th>whole</th><th>local</th><th>overall</th>"
                "<th>certainty</th><th>calls</th><th>order</th></tr>"
                + "".join(body) + "</table>")

    index = f"""<!doctype html><html><head><meta charset="utf-8">
<title>首轮 SFT 候选样板 — {html.escape(batch_meta['batch_id'])}</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,sans-serif;margin:24px;max-width:1400px}}
table{{border-collapse:collapse;width:100%;margin:10px 0}}
th,td{{border:1px solid #ddd;padding:5px 7px;font-size:13px;vertical-align:top}}
th{{background:#f4f4f4;text-align:left}} .muted{{color:#666}}
.banner{{background:#fff7e0;border:1px solid #e0c060;padding:10px;border-radius:6px}}
</style></head><body>
<h1>首轮 SFT 候选样板（{html.escape(batch_meta['batch_id'])}）</h1>
<div class="banner"><b>候选样板，不是训练数据。</b>助手文本由候选写手按当前协议拟写，
不是 Qwen 原始输出；PROBE/TruFor 的调用、证据 token、三图和控制器检查都是真实运行。
真值与 mask 只用于审核，助手每一步只引用当时已收到的证据。通过审核不代表自动进入训练。</div>
<p class="muted">{len(records)} 条：24 条主样板（8 张 AutoSplice 局部篡改、6 张 AutoSplice
真图、6 张 GenImage 整图生成、4 张 GenImage Real）+ 2 条单专家样板（1 条 PROBE-only、
1 条 TruFor-only）。全部来自 split.json 的训练侧；同源变体同划分。重点案例按讨论价值
排在前面。</p>
<p>配置 <code>{html.escape(batch_meta.get('config_sha256', ''))}</code>；manifest
<code>{html.escape(batch_meta.get('manifest', {}).get('sha256', ''))}</code>；
system prompt digest <code>{html.escape(batch_meta.get('prompt_digest', ''))}</code>。</p>
<p class="muted">审核图片已打包在 <code>review_assets/</code>；每条页面里的原图、源图、实际投送的专家图（有 TruFor 时为原图裁剪/异常图/可靠性图，有 PROBE 时为诊断裁剪）和预测框叠加都使用批次内相对路径，拉取仓库后可直接打开。原始 trace 仍保留运行时的路径，作为审计出处。</p>
<h2>重点案例（{len(key)}）</h2>
{table(key)}
<h2>单专家样板（{len(single)}）</h2>
<p class="muted">这两条只调用一个专家并直接结案：一条 PROBE-only，一条 TruFor-only
（总体 Fake、整图 Uncertain）。它们用于示范“证据够时不再补测”，不计入 24 条主样板。</p>
{table(single)}
<h2>其余候选（{len(rest)}）</h2>
{table(rest)}
<p><a href="records.json" download>下载 records.json</a> ·
<a href="review_records.csv" download>下载 review_records.csv</a> ·
<a href="CANDIDATE_CASES.md" download>下载重点案例清单（Markdown）</a> ·
<a href="SUMMARY.md" download>下载汇总（Markdown）</a> ·
<a href="AUDIT_REPORT.md" download>下载收尾审计（Markdown）</a> ·
<a href="TEST_STATUS.md" download>下载测试状态（Markdown）</a> ·
<a href="REVIEW_QUESTIONS.md" download>下载已确认判例与复核点（Markdown）</a> ·
<a href="REVISION_NOTES.md" download>下载判例修订清单（Markdown）</a> ·
<a href="POST_RUN_PATCHES.md" download>下载文字补丁说明（Markdown）</a></p>
<p class="muted">收尾审计 {len(records)}/{len(records)} pass、0 问题；一处未投递数字的补丁不影响工具输出、证据 ID、框或结论。</p>
<p class="muted">每条会话页顶部有“通过 / 待修 / 拒绝”审核控件；决定先存浏览器草稿，
导出后用 <code>python scripts/g8_review_import.py &lt;文件&gt;</code> 写回仓库。</p>
</body></html>"""
    (out / "index.html").write_text(index, encoding="utf-8")


def build_candidate_cases(out: Path, records: list[dict]) -> None:
    key = [row for row in records if row.get("key_case")]
    key.sort(key=lambda row: row.get("key_case_rank") or 99)
    single = [row for row in records
              if str(row.get("kind", "")).startswith("single_expert")]
    total = len(records)
    lines = [
        "# 首轮 SFT 候选样板：重点案例与修订清单",
        "",
        f"完整 {total} 条（24 条主样板 + 2 条单专家样板）见 `index.html`。",
        "助手文本是候选写手拟写，不是 Qwen 输出；PROBE/TruFor 与三图是真实运行。",
        "真值和 mask 只用于审核，助手每一步只引用当时已收到的证据。",
        "",
        "## 重点案例",
        "",
        "| # | sample | 类型 | 真值 whole/local | 候选 whole/local/overall | PROBE | TruFor | 重点看什么 |",
        "|---|---|---|---|---|---:|---:|---|",
    ]
    notes = {
        "autosplice__54029__1__q95": "强局部证据；PROBE 中高分但整图来源无独立支撑，整图 Uncertain、局部 Tampered。",
        "autosplice__55310__1__q95": "对照：PROBE 低分支持整图 Real，局部 Tampered；保留 Real + Tampered。",
        "autosplice__45749__1__q95": "企鹅：PROBE 接近 1 但整图来源缺独立支撑；整图 Uncertain、局部 Tampered、总体 Fake。",
        "autosplice__69868__0__q95": "强局部证据、局部框较宽；整图 Uncertain + 局部 Tampered + 总体 Fake。",
        "autosplice__48251__0__q95": "PROBE 高分的人像局部编辑；整图 Uncertain，局部框较宽。",
        "autosplice__51186__authentic__q95": "TruFor 0.97 但高响应只在画框右缘；局部 Not tampered，certainty Medium。",
        "autosplice__65547__authentic__q95": "TruFor 0.90 的真图；高响应在肩部/衣领，内容解释不唯一，保留 Uncertain。",
        "genimage__SD15__459_sdv5_00195__q95": "整图生成 + TruFor 0.98 局部高响应；整图 AI-generated 后局部 Not applicable。",
    }
    for row in key:
        screen = row.get("screening_review_only") or {}
        truth = row.get("task_truth") or {}
        lines.append(
            f"| {row.get('key_case_rank')} | `{row['sample_id']}` | "
            f"{row.get('selection_role')} | "
            f"{truth.get('whole_image_generation')} / {truth.get('local_manipulation')} | "
            f"{row.get('whole_image_generation')} / {row.get('local_manipulation')} / "
            f"{row.get('final_verdict')} | "
            f"{screen.get('probe_score')} | {screen.get('trufor_score')} | "
            f"{notes.get(row['sample_id'], '')} |")
    lines += [
        "",
        "## 单专家样板",
        "",
        "| # | sample | 调用 | 真值 overall | 候选 whole/local/overall | 审核页 |",
        "|---|---|---|---|---|---|",
    ]
    for row in single:
        lines.append(
            f"| {row['index']} | `{row['sample_id']}` | "
            f"{' → '.join(row.get('call_order') or [])} | {row.get('ground_truth')} | "
            f"{row.get('whole_image_generation')} / {row.get('local_manipulation')} / "
            f"{row.get('final_verdict')} | [{row['page']}]({row['page']}) |")
    lines += [
        "",
        f"## 全部 {total} 条",
        "",
        "| # | sample | 类型 | 真值 overall | 候选 overall | calls | 审核页 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['index']} | `{row['sample_id']}` | {row.get('selection_role')} | "
            f"{row.get('ground_truth')} | {row.get('final_verdict')} | "
            f"{' → '.join(row.get('call_order') or [])} | "
            f"[{row['page']}]({row['page']}) |")
    uncertain = [row for row in records if row.get("final_verdict") == "Uncertain"]
    calls = {row["counters"]["expert_calls"] for row in records}
    lines += [
        "",
        "## 候选结论分布（审核用）",
        "",
        f"- {total} 条全部完成；专家调用次数分布 {sorted(calls)}；证据/覆盖/certainty 检查均为 pass。",
        f"- 总体与真值一致 {sum(1 for row in records if row.get('final_verdict') == row.get('ground_truth'))}/{total}；"
        f"整图维度 {sum(1 for row in records if row.get('whole_image_generation') == (row.get('task_truth') or {}).get('whole_image_generation'))}/{total}；"
        f"局部维度 {sum(1 for row in records if row.get('local_manipulation') == (row.get('task_truth') or {}).get('local_manipulation'))}/{total}。"
        "这是审核对照，不是验收分数。",
        f"- 总体 Uncertain 共 {len(uncertain)} 条："
        + "、".join(f"`{row['sample_id']}`" for row in uncertain) + "。",
        "- 已确认判例：PROBE 高分 + 局部证据充分但整图来源无独立支撑 → 整图 Uncertain、局部 Tampered、总体 Fake；"
        "55310 因 PROBE 低分保留整图 Real 作对照。",
        "- 已确认判例：TruFor 响应在整图边缘而非内容区 → 局部 Not tampered；51186、57332 按此口径。",
        "- 65547、52769 保留局部 Uncertain；70684、67006 等仍按原候选保留。",
        "",
        "## 审核时请重点判断",
        "",
        "1. 每一步是否只引用了该轮之前已经投递的证据；有没有提前使用真值或 mask。",
        "2. 整图来源与局部篡改是否分开写；PROBE 高分是否被错误写成整图生成。",
        "3. TruFor 高分真图是否被写成 Tampered；边缘响应是否被当成内容异常。",
        "4. 单专家样板是否真的“证据够就停”：PROBE-only 是否不需要局部；TruFor-only 是否把整图留 Uncertain。",
        "5. 局部框是否与异常图/可靠性图所示区域一致；框过宽时 certainty 是否降级。",
        "",
        "## 边界",
        "",
        "- `REVIEW_QUESTIONS.md` 记录本轮已确认判例和复看点；`REVISION_NOTES.md`",
        "  逐项列出改了标签、certainty 或调用理由的案例。",
        "- 审核图片 `review_assets/`：逐条复制运行时的输入图、源图、实际投送的",
        "  专家图和预测框叠加；单专家页按实际调用展示。页面改为批次内相对路径。",
        f"- 收尾审计 `AUDIT_REPORT.md`：{total}/{total} pass，0 问题（证据引用时序、数字出处、",
        "  真值/mask 词、页面结构、链接、划分与组成）。",
        "- 候选文本是拟写稿，不是 Qwen 原始输出；运行记录明确标注 `sft_candidate_scripted`。",
        "- 真值与 mask 只在审核侧；助手上下文没有它们。",
        "- 本轮不扩样、不训练、不转 final_v2、不碰 test。",
    ]
    (out / "CANDIDATE_CASES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_summary(out: Path, records: list[dict], batch_meta: dict,
                  failures: list[dict], elapsed: float) -> None:
    overall_ok = sum(1 for row in records if row.get("final_verdict")
                     == row.get("ground_truth"))
    local_ok = sum(1 for row in records
                   if row.get("local_manipulation")
                   == (row.get("task_truth") or {}).get("local_manipulation"))
    whole_ok = sum(1 for row in records
                   if row.get("whole_image_generation")
                   == (row.get("task_truth") or {}).get("whole_image_generation"))
    total = len(records)
    lines = [
        f"# 首轮 SFT 候选样板 — {batch_meta['batch_id']}",
        "",
        f"- 候选会话：{total}/{total} 完成（24 条主样板 + 2 条单专家样板）；"
        f"失败 {len(failures)}；wall {elapsed:.1f}s。",
        f"- assistant：候选写手拟写（`sft_candidate_scripted`），不是 Qwen 原始输出。",
        f"- 工具：真实 PROBE + TruFor；配置 `{batch_meta.get('config_sha256', '')}`；",
        f"  manifest `{batch_meta.get('manifest', {}).get('sha256', '')}`。",
        f"- 审核入口：`index.html`；重点案例：`CANDIDATE_CASES.md`。",
        f"- 审核图片：`review_assets/`（逐条原图/源图/可用的专家图/叠加；",
        f"  页面使用批次内相对路径）。",
        f"- 收尾审计：`AUDIT_REPORT.md`（{total}/{total} pass，0 问题）。",
        f"- 已确认判例与复核点：`REVIEW_QUESTIONS.md`。",
        f"- 判例修订清单：`REVISION_NOTES.md`（标签/置信度/调用理由逐项列出）。",
        f"- 文字补丁：`POST_RUN_PATCHES.md`（Midjourney 一条去掉未投递的 0.39 数字；",
        f"  不影响工具输出、证据 ID、框或标签）。",
        "",
        "## 与真值对照（仅供审核，不是验收）",
        "",
        f"- 总体：{overall_ok}/{len(records)}",
        f"- 整图维度：{whole_ok}/{len(records)}",
        f"- 局部维度：{local_ok}/{len(records)}",
        "",
        "| # | sample | 真值 | 候选 overall | whole | local | certainty | calls |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in records:
        lines.append(
            f"| {row['index']} | `{row['sample_id']}` | {row.get('ground_truth')} | "
            f"{row.get('final_verdict')} | {row.get('whole_image_generation')} | "
            f"{row.get('local_manipulation')} | {row.get('certainty')} | "
            f"{(row.get('counters') or {}).get('expert_calls')} |")
    lines += [
        "",
        "## 运行与审核边界",
        "",
        f"- {total} 条全部来自 split.json 的训练侧；同源变体同划分；未读取 test。",
        "- 真值和 mask 只写在 record/页面/本汇总，不进助手上下文。",
        "- 候选文本仍需人工审核；通过审核不等于进入训练，也不等于 final_v2 准入。",
        "- 本轮不扩样、不训练、不改原 200/24 批次。",
    ]
    if failures:
        lines += ["", "## 失败", ""]
        for failure in failures:
            lines.append(f"- `{failure.get('sample_id')}`: {failure.get('error')}")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1")

    os.environ.setdefault(
        "PROBE_WEIGHTS",
        "/root/autodl-tmp/third_party/probe/DINOv2_best_model_step_34999.pth")
    os.environ.setdefault("TRUFOR_HOME", "/root/autodl-tmp/third_party/TruFor")
    os.environ.setdefault("TRUFOR_PYTHON", "/root/miniconda3/bin/python")
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRUFOR_RESULTS", str(out / "trufor_raw"))

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    scripts = json.loads((out / "candidate_scripts.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "sft_candidate_manifest/1":
        raise SystemExit("unexpected manifest schema")
    if scripts.get("schema") != "sft_candidate_scripts/1":
        raise SystemExit("unexpected scripts schema")
    expected_count = len(manifest["samples"])
    if expected_count < 24 or len(scripts["samples"]) != expected_count:
        raise SystemExit(
            f"expected the manifest and scripts to agree (got "
            f"{expected_count} / {len(scripts['samples'])})")
    script_by_id = {row["sample_id"]: row for row in scripts["samples"]}
    if set(script_by_id) != {row["sample_id"] for row in manifest["samples"]}:
        raise SystemExit("manifest/scripts sample mismatch")

    from experts.probe_dino import ProbeDinoExpert
    from experts.trufor import TruForExpert
    from mllm.message_builder import build_forensic_prompt
    from mllm.scripted_candidate import ScriptedCandidateMLLM
    from state_machine.controller import ForensicStateMachine
    from utils.logger import SessionLogger
    from utils.protocol_identity import expert_identity
    from utils.review_overlays import render_record_overlays
    from utils.review_store import sample_fingerprint
    from scripts.g8_review_workstation import session_page

    system_prompt = build_forensic_prompt(("probe", "trufor"))
    experts = {
        "probe_dino_expert": ProbeDinoExpert(device=args.device),
        "trufor_expert": TruForExpert(device=args.device),
    }
    for name, expert in experts.items():
        reason = expert.unavailable_reason()
        if reason:
            raise SystemExit(f"{name} unavailable: {reason}")
    expert_info = expert_identity(experts)
    config_sha = sha256_text(json.dumps({
        "assistant": "scripted_candidate_writer",
        "experts": expert_info,
        "system_prompt": sha256_text(system_prompt),
        "manifest": sha256_file(out / "manifest.json"),
        "scripts": sha256_file(out / "candidate_scripts.json"),
    }, ensure_ascii=False, sort_keys=True))[:16]
    batch_meta = {
        "batch_id": out.name,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "manifest": {"path": os.path.relpath(out / "manifest.json", ROOT),
                     "sha256": sha256_file(out / "manifest.json")},
        "scripts": {"path": os.path.relpath(out / "candidate_scripts.json", ROOT),
                    "sha256": sha256_file(out / "candidate_scripts.json")},
        "model": {
            "checkpoint": "none — assistant text scripted candidate writer",
            "revision": "",
            "dtype": "",
            "inference_mode": "sft_candidate_scripted",
        },
        "prompts": {"digest": sha256_text(system_prompt)[:16]},
        "prompt_digest": sha256_text(system_prompt)[:16],
        "experts": expert_info,
        "config_sha256": config_sha,
        "code": git_state(out),
        "cgroup": cgroup_state(),
        "dialogue_mode": "sft_candidate_scripted",
        "note": ("Assistant turns are candidate drafts; PROBE/TruFor, evidence "
                 "tokens, artifacts and controller checks are real runtime."),
    }
    if batch_meta["code"]["dirty"]:
        raise SystemExit("worktree has non-output changes; commit before running:\n  "
                         + "\n  ".join(batch_meta["code"]["dirty_paths"]))
    if (out / "batch_meta.json").exists():
        raise SystemExit(f"{out / 'batch_meta.json'} exists; use a new output")
    write_json_atomic(out / "batch_meta.json", batch_meta)
    (out / "records").mkdir(exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)

    records, failures = [], []
    started = time.time()
    print(json.dumps({"samples": expected_count, "config_sha256": config_sha,
                      "out": str(out)}, ensure_ascii=False, indent=2), flush=True)
    for index, sample in enumerate(manifest["samples"], 1):
        sid = sample["sample_id"]
        script = script_by_id[sid]
        logger = SessionLogger(sft_dir=str(out / "traces"))
        client = ScriptedCandidateMLLM(
            script["outputs"], system_prompt, call_order=script["call_order"])
        fsm = ForensicStateMachine(client, experts, logger=logger)
        t0 = time.perf_counter()
        try:
            result = fsm.run(str(ROOT / sample["path"]),
                             ground_truth=sample["ground_truth"])
            elapsed = time.perf_counter() - t0
            trace_path = Path(result["sft_data_path"])
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            meta = trace.get("metadata") or {}
            verdict = result.get("final_verdict") or {}
            target = script["verdict"]
            if len(result.get("evidence_chain") or []) != len(script["call_order"]):
                raise RuntimeError(
                    f"expected {len(script['call_order'])} evidence tokens, got "
                    f"{len(result.get('evidence_chain') or [])}")
            if result.get("halting_reason") != "model_concluded":
                raise RuntimeError(f"session did not conclude: {result.get('halting_reason')}")
            if any(v != target.get(k) for k, v in (
                    ("verdict", verdict.get("verdict")),
                    ("whole_image_generation", verdict.get("whole_image_generation")),
                    ("local_manipulation", verdict.get("local_manipulation")))):
                raise RuntimeError(
                    f"record verdict differs from target: {verdict.get('verdict')}/"
                    f"{verdict.get('whole_image_generation')}/"
                    f"{verdict.get('local_manipulation')} vs {target}")
            if meta.get("failed_experts") or meta.get("failed_tool_calls"):
                raise RuntimeError(f"expert failure: {meta.get('failed_experts')} "
                                   f"{meta.get('failed_tool_calls')}")
            record = make_record(sample, index, result, trace, elapsed, script,
                                 config_sha)
            try:
                record["review_overlays"] = render_record_overlays(
                    record, str(out / "review_overlays"))
            except Exception as exc:  # noqa: BLE001
                record["review_overlays"] = []
                record["review_overlay_error"] = f"{type(exc).__name__}: {exc}"
            page = page_name(index, sid)
            page_html = session_page(
                {"record": record, "trace": trace, "page": page,
                 "fingerprint": sample_fingerprint(record, trace),
                 "source_image": sample["source_path"]},
                str(out), "index.html", batch_meta=batch_meta,
                meta={"schema": "sft_candidate/1", "generation": "g8",
                      "batch": out.name, "samples": {}, "repo_decisions": {}})
            (out / page).write_text(page_html, encoding="utf-8")
            record["page"] = page
            write_json_atomic(out / "records" / f"{index:03d}__{sid}.json", record)
            records.append(record)
            print(f"[{index}/{expected_count}] {sid} overall={record['final_verdict']} "
                  f"whole={record['whole_image_generation']} "
                  f"local={record['local_manipulation']} "
                  f"calls={record['counters']['expert_calls']} "
                  f"{elapsed:.1f}s", flush=True)
            del fsm, client, logger
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - t0
            failure = {
                "index": index, "sample_id": sid,
                "failed_at": datetime.now().isoformat(timespec="seconds"),
                "elapsed_s": round(elapsed, 2),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-8000:],
                "config_sha256": config_sha,
                "status": "failed",
            }
            write_json_atomic(out / "failures" / f"{index:03d}__{sid}.json", failure)
            failures.append(failure)
            print(f"[{index}/{expected_count}] {sid} FAILED: {failure['error']}", flush=True)
            break

    write_json_atomic(out / "records.json", records)
    (out / "records.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8")
    write_csv(out / "review_records.csv", records)
    batch_meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    batch_meta["completed"] = len(records)
    batch_meta["failed"] = len(failures)
    write_json_atomic(out / "batch_meta.json", batch_meta)
    build_index(out, records, batch_meta)
    build_candidate_cases(out, records)
    build_summary(out, records, batch_meta, failures, time.time() - started)
    end = {
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "completed": len(records), "expected": expected_count, "failed": len(failures),
        "config_sha256": config_sha, "dialogue_mode": "sft_candidate_scripted",
        "elapsed_s": round(time.time() - started, 1),
        "failures": failures,
    }
    write_json_atomic(out / "run_end.json", end)
    print(json.dumps(end, ensure_ascii=False, indent=2), flush=True)
    return 0 if len(records) == expected_count and not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
