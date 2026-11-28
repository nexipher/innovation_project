"""Audit the 24 first-round SFT candidate trajectories without a GPU.

Checks, independently of the controller's own ledger:
  * every evidence id cited by an assistant turn was delivered before that turn;
  * every decimal number is a delivered score/map statistic (or the 0.5
    threshold), and an expert score is not quoted before that expert ran;
  * ground-truth/mask words do not appear in any assistant turn;
  * the verdict JSON, record and target agree;
  * every page has the original, the three TruFor panels, the PROBE crop, the
    score table, the call-order row and the closing fields; links resolve;
  * split/composition/key-case invariants hold.

Writes AUDIT_REPORT.md and audit.json.  Read-only otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "sft_data/sft_candidates_20261127"
FORBIDDEN = ("交叉验证", "相互验证", "共同支持", "相互支持")
TRUTH_WORDS = ("真值", "ground truth", "ground_truth", "mask", "掩码")
SCORE_RE = re.compile(r"(PROBE|probe|TruFor|trufor)\s*(?:分数|score)?\s*[:：]?\s*(0\.\d+)")
DECIMAL_RE = re.compile(r"0\.\d+")
EID_RE = re.compile(r"E-[0-9a-f]{10}")
VERDICT_RE = re.compile(r"<verdict>\s*(\{.*?\})\s*</verdict>", re.S)


def _score(token: dict) -> float | None:
    value = (token.get("score") or {}).get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _map_values(token: dict) -> list[float]:
    stats = token.get("map_stats") or {}
    values = []
    for value in stats.values():
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _match_number(text: str, candidates: list[float]) -> bool:
    """A decimal in prose matches a candidate at 2-6 decimals."""
    number = float(text)
    for candidate in candidates:
        for digits in (2, 3, 4, 5, 6):
            if round(candidate, digits) == round(number, digits):
                return True
    return False


def audit_trace(trace: dict, record: dict, script: dict) -> list[str]:
    issues = []
    delivered: list[dict] = []
    conversations = trace.get("conversations") or []
    for index, turn in enumerate(conversations):
        value = turn.get("value") or ""
        if turn.get("from") != "gpt":
            if value.lstrip().startswith("{"):
                try:
                    payload = json.loads(value)
                except ValueError:
                    payload = None
                if isinstance(payload, dict) and payload.get("evidence_id"):
                    delivered.append(payload)
            continue

        delivered_ids = {token.get("evidence_id") for token in delivered}
        for eid in EID_RE.findall(value):
            if eid not in delivered_ids:
                issues.append(f"turn {index}: cites undelivered evidence id {eid}")
        for phrase in FORBIDDEN:
            if phrase in value:
                issues.append(f"turn {index}: forbidden phrase {phrase!r}")
        for word in TRUTH_WORDS:
            if word in value:
                issues.append(f"turn {index}: truth/mask word {word!r} in assistant text")

        by_expert = {}
        for token in delivered:
            by_expert.setdefault(token.get("expert"), []).append(token)
        for match in SCORE_RE.finditer(value):
            expert = "probe_dino_expert" if match.group(1).lower() == "probe" else "trufor_expert"
            if expert not in by_expert:
                issues.append(
                    f"turn {index}: quotes {expert} score before that expert ran")
                continue
            tokens = by_expert[expert]
            if not any(_score(token) is not None
                       and _match_number(match.group(2), [_score(token)])
                       for token in tokens):
                issues.append(
                    f"turn {index}: {expert} score {match.group(2)} does not match "
                    f"delivered token")

        allowed = [0.5]
        for token in delivered:
            score = _score(token)
            if score is not None:
                allowed.append(score)
            allowed.extend(_map_values(token))
        for number in DECIMAL_RE.findall(value):
            if not _match_number(number, allowed):
                issues.append(
                    f"turn {index}: decimal {number} is not a delivered score/stat")

    final = conversations[-1].get("value") if conversations else ""
    match = VERDICT_RE.search(final or "")
    if not match:
        issues.append("final turn has no parseable verdict JSON")
    else:
        try:
            verdict = json.loads(match.group(1))
        except ValueError:
            verdict = {}
        if not isinstance(verdict, dict):
            issues.append("final verdict is not an object")
            verdict = {}
        for key in ("verdict", "whole_image_generation", "local_manipulation"):
            if verdict.get(key) != record.get(
                    "final_verdict" if key == "verdict" else key):
                issues.append(f"final verdict {key} differs from record")

    if trace.get("metadata", {}).get("mock_mode") != "sft_candidate_scripted":
        issues.append("trace mock_mode is not sft_candidate_scripted")
    if len(delivered) != len(script["call_order"]):
        issues.append(f"delivered {len(delivered)} tokens, expected {len(script['call_order'])}")
    if len([turn for turn in conversations if turn.get("from") == "gpt"]) != len(script["outputs"]):
        issues.append("assistant turn count differs from script")
    return issues


def audit_page(page: Path, record: dict) -> list[str]:
    issues = []
    if not page.is_file():
        return [f"missing page {page.name}"]
    text = page.read_text(encoding="utf-8")
    for label in ("图像与版本", "本次得到的专家原始分数", "已记录的会话原文",
                  "控制与计数", "结案来源", "局部篡改判断", "整图生成判断",
                  "候选调用顺序"):
        if label not in text:
            issues.append(f"page missing section/label {label!r}")
    for panel in ("model_original_crop", "model_anomaly_map",
                  "model_confidence_map"):
        if panel not in text:
            issues.append(f"page missing {panel}")
    if "region_E-" not in text:
        issues.append("page missing PROBE diagnostic crop")
    if record.get("page") and record["page"] not in text:
        pass  # page does not need to link itself
    if "候选样板：" not in text:
        issues.append("page missing scripted-candidate banner")
    if " → ".join(record.get("call_order") or []) not in text:
        issues.append("page missing the call order")
    for expert in ("probe_dino_expert", "trufor_expert"):
        if expert not in text:
            issues.append(f"page missing {expert} in evidence table")
    # Link targets.
    for link in re.findall(r'(?:src|href)="([^"]+)"', text):
        if link.startswith("http") or link.startswith("#"):
            continue
        if not (page.parent / link).resolve().exists():
            issues.append(f"broken link {link}")
    return issues


def audit_split(out: Path) -> list[str]:
    issues = []
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if len(samples) != 24:
        issues.append(f"manifest has {len(samples)} samples")
    if Counter(row["kind"] for row in samples) != Counter({
            "autosplice_forged": 8, "autosplice_authentic": 6,
            "genimage_generated": 6, "genimage_real": 4}):
        issues.append(f"composition differs: {dict(Counter(row['kind'] for row in samples))}")
    if any(row.get("split") != "train" for row in samples):
        issues.append("some samples are not on the train side")
    auto_sources = [row["source_id"] for row in samples
                    if row["kind"].startswith("autosplice")]
    if len(auto_sources) != len(set(auto_sources)):
        issues.append("AutoSplice base reused")
    gen_sources = [row["source_id"] for row in samples
                   if not row["kind"].startswith("autosplice")]
    if len(gen_sources) != len(set(gen_sources)):
        issues.append("GenImage/Real source reused")
    key = [row for row in samples if row.get("key_case")]
    if len(key) != 8:
        issues.append(f"expected 8 key cases, got {len(key)}")
    if samples[:8] != key:
        issues.append("key cases are not the first eight manifest rows")
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    scripts = {row["sample_id"]: row for row in json.loads(
        (out / "candidate_scripts.json").read_text(encoding="utf-8"))["samples"]}
    records = json.loads((out / "records.json").read_text(encoding="utf-8"))
    by_id = {row["sample_id"]: row for row in records}

    global_issues = audit_split(out)
    rows = []
    total_issues = 0
    for sample in manifest["samples"]:
        sid = sample["sample_id"]
        record = by_id.get(sid)
        if record is None:
            rows.append({"sample_id": sid, "issues": ["missing record"]})
            total_issues += 1
            continue
        trace_path = ROOT / record["trace_path"]
        if not trace_path.is_file():
            rows.append({"sample_id": sid, "issues": ["missing trace"]})
            total_issues += 1
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        issues = audit_trace(trace, record, scripts[sid])
        issues += audit_page(out / record["page"], record)
        total_issues += len(issues)
        rows.append({
            "sample_id": sid,
            "key_case_rank": record.get("key_case_rank"),
            "assistant_turns": len([t for t in trace.get("conversations", [])
                                    if t.get("from") == "gpt"]),
            "evidence": len(record.get("evidence") or []),
            "issues": issues,
        })

    payload = {
        "schema": "sft_candidate_audit/1",
        "samples": len(rows),
        "issues": total_issues,
        "global_issues": global_issues,
        "rows": rows,
    }
    (out / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# 首轮 SFT 候选样板 — 收尾审计",
        "",
        f"- 审计样本：{len(rows)}/24；发现问题：{total_issues}；全局问题：{len(global_issues)}。",
        "- 审计不加载模型、不读 GPU；只读 manifest/split/records/traces/页面。",
        "- 检查项：证据引用时序、数字与已投递 token 对齐、真值/mask 词、结论字段、",
        "  页面原图/三图/分数表/调用顺序/结案/链接、划分与组成。",
        "",
        "## 全局",
        "",
        "\n".join(f"- {issue}" for issue in global_issues) or "- 无。",
        "",
        "## 逐条",
        "",
        "| # | sample | assistant turns | evidence | 结果 |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        status = "pass" if not row["issues"] else " / ".join(row["issues"])
        lines.append(
            f"| {row.get('key_case_rank') or ''} | `{row['sample_id']}` | "
            f"{row.get('assistant_turns')} | {row.get('evidence')} | {status} |")
    lines += [
        "",
        "## 审计边界",
        "",
        "- 数字检查覆盖 0.x 形式；它验证“出处”，不验证文字解释是否最优。",
        "- 真值/mask 词检查是词表检查；未命中不等于已证明没有泄露。",
        "- 页面检查是结构检查；内容质量仍需人工审核。",
        "- 本审计不改变任何候选轨迹，也不把候选变成训练数据。",
    ]
    (out / "AUDIT_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(rows), "issues": total_issues,
                      "global_issues": global_issues,
                      "out": str(out / "AUDIT_REPORT.md")},
                     ensure_ascii=False, indent=2))
    return 0 if total_issues == 0 and not global_issues else 1


if __name__ == "__main__":
    raise SystemExit(main())
