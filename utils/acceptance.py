"""
逐样本验收核对（2026-10-18 用户指定，plan §4.12b）。

门禁不能"数决定条数"：台账里其他样本的决定会凑数，重复 ID、空批次、损坏记录
也都能蒙混过去。这里以**记录目录里的真实 `trajectory_id`** 为准：

  * 记录：目录必须存在且非空；每个 `.json`（除 `metadata.json`）必须能解析且
    带 `trajectory_id`；跨文件重复 ID 拒绝；期望样本集（b2 的变体清单 +
    metadata 里的 policy）与实际记录逐 ID 比对，少一条/多一条都拒绝。
  * trace：每条记录必须有 `trace_path`，文件存在且能解析；用工作台同一个
    `sample_fingerprint(record, trace)` **现场重算**指纹。
  * 台账：文件必须存在且非空；每行必须能解析，且 `generation/batch` 与当前
    一致；只有**当前样本的最新决定**参与覆盖。
  * 判定：当前每个 ID 都必须有最新 `pass`，且决定里的指纹等于现场指纹。
    台账的 `stale` 字段只是导入时的旧判断，**不作为放行依据**（也不改写它）。

任何一项失败都返回具体 ID 与原因；调用方（外部 Agent 门禁、验收身份登记）
据此拒绝。这里**不修改**任何决定、不自动改成通过、不重写历史。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from config import PROJECT_ROOT
from utils.review_store import Decision, latest, sample_fingerprint

#: 不算会话记录的文件
NON_RECORD_FILES = ("metadata.json",)


@dataclass
class Problem:
    """一个拒绝理由。`trajectory_id` 为空表示批次级问题。"""

    kind: str
    detail: str
    trajectory_id: str = ""

    def render(self) -> str:
        return f"{self.detail}"


def _rel(path: str) -> str:
    try:
        return os.path.relpath(path, PROJECT_ROOT)
    except ValueError:
        return path


# ---------------------------------------------------------------------------
# 读取
# ---------------------------------------------------------------------------

def load_records_strict(directory: str) -> Tuple[Dict[str, dict], List[Problem]]:
    """
    严格读会话记录：不静默跳过任何 `.json`。

    返回值是 `trajectory_id -> record`；任何目录级/文件级问题都进 problems。
    """
    problems: List[Problem] = []
    if not os.path.isdir(directory):
        return {}, [Problem("records_missing", f"记录目录不存在：{_rel(directory)}")]
    names = sorted(name for name in os.listdir(directory)
                   if name.endswith(".json") and name not in NON_RECORD_FILES)
    if not names:
        return {}, [Problem("empty_batch",
                            f"批次为空：{_rel(directory)} 里没有会话记录")]

    records: Dict[str, dict] = {}
    origin: Dict[str, str] = {}
    for name in names:
        path = os.path.join(directory, name)
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError) as exc:
            problems.append(Problem("record_unreadable",
                                    f"{name}: 无法解析（{type(exc).__name__}: {exc}）"))
            continue
        if not isinstance(record, dict) or not record.get("trajectory_id"):
            problems.append(Problem("record_not_trajectory",
                                    f"{name}: 不是轨迹记录（缺 trajectory_id）"))
            continue
        trajectory_id = str(record["trajectory_id"])
        if trajectory_id in records:
            problems.append(Problem(
                "duplicate_id",
                f"{trajectory_id}: 重复记录 ID（{origin[trajectory_id]} 与 {name}）",
                trajectory_id))
            continue
        records[trajectory_id] = record
        origin[trajectory_id] = name
    return records, problems


def load_traces(records: Dict[str, dict]) -> Tuple[Dict[str, dict], Dict[str, str], List[Problem]]:
    """读每条记录的 trace，并现场重算内容指纹。"""
    traces: Dict[str, dict] = {}
    fingerprints: Dict[str, str] = {}
    problems: List[Problem] = []
    for trajectory_id, record in sorted(records.items()):
        trace_path = record.get("trace_path")
        if not trace_path:
            problems.append(Problem("trace_not_recorded",
                                    f"{trajectory_id}: 记录里没有 trace_path",
                                    trajectory_id))
            continue
        absolute = os.path.join(PROJECT_ROOT, trace_path)
        if not os.path.isfile(absolute):
            problems.append(Problem("trace_missing",
                                    f"{trajectory_id}: trace 文件不存在：{trace_path}",
                                    trajectory_id))
            continue
        try:
            with open(absolute, encoding="utf-8") as handle:
                trace = json.load(handle)
        except (OSError, ValueError) as exc:
            problems.append(Problem(
                "trace_unreadable",
                f"{trajectory_id}: trace 无法解析：{trace_path}"
                f"（{type(exc).__name__}: {exc}）", trajectory_id))
            continue
        if not isinstance(trace, dict):
            problems.append(Problem("trace_not_object",
                                    f"{trajectory_id}: trace 不是对象：{trace_path}",
                                    trajectory_id))
            continue
        try:
            fingerprints[trajectory_id] = sample_fingerprint(record, trace)
        except Exception as exc:                    # 指纹算不出来 = 不能核对
            problems.append(Problem(
                "fingerprint_failed",
                f"{trajectory_id}: 无法计算内容指纹（{type(exc).__name__}: {exc}）",
                trajectory_id))
            continue
        traces[trajectory_id] = trace
    return traces, fingerprints, problems


def load_ledger_strict(generation: str, batch: str,
                       path: str) -> Tuple[List[Decision], List[Problem]]:
    """
    严格读台账：缺文件、空文件、坏行、跨代次/跨批次的记录都拒绝。

    同一样本的多行是**追加历史**，由 `latest()` 取最后一条；这里不折叠。
    """
    problems: List[Problem] = []
    if not os.path.exists(path):
        return [], [Problem("ledger_missing", f"验收台账不存在：{_rel(path)}")]
    decisions: List[Decision] = []
    try:
        handle = open(path, encoding="utf-8")
    except OSError as exc:
        return [], [Problem("ledger_unreadable",
                            f"验收台账无法读取：{_rel(path)}"
                            f"（{type(exc).__name__}: {exc}）")]
    with handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except ValueError as exc:
                problems.append(Problem(
                    "ledger_unreadable",
                    f"{_rel(path)} 第 {number} 行无法解析（{exc}）"))
                continue
            if not isinstance(raw, dict):
                problems.append(Problem(
                    "ledger_unreadable",
                    f"{_rel(path)} 第 {number} 行不是对象"))
                continue
            if raw.get("generation") != generation or raw.get("batch") != batch:
                problems.append(Problem(
                    "ledger_wrong_batch",
                    f"{_rel(path)} 第 {number} 行属于 "
                    f"{raw.get('generation')}/{raw.get('batch')}，"
                    f"不是 {generation}/{batch}"))
                continue
            decisions.append(Decision(
                trajectory_id=str(raw.get("trajectory_id") or ""),
                fingerprint=str(raw.get("fingerprint") or ""),
                decision=str(raw.get("decision") or ""),
                reviewer=str(raw.get("reviewer") or ""),
                category=raw.get("category"),
                note=str(raw.get("note") or ""),
                decided_at=str(raw.get("decided_at") or ""),
                stale=bool(raw.get("stale")),
                stale_reason=str(raw.get("stale_reason") or ""),
            ))
    if not decisions and not problems:
        problems.append(Problem("ledger_empty", f"验收台账为空：{_rel(path)}"))
    return decisions, problems


def expected_ids_from_batch(generation: str, batch: str,
                            records_dir: str,
                            manifest_path: Optional[str] = None,
                            metadata_path: Optional[str] = None
                            ) -> Tuple[Optional[Set[str]], List[Problem]]:
    """
    期望样本集：变体清单里的每个 `variant_id` → `{policy}__{variant_id}`。

    没有这一层，"记录被删掉几条"会被当成"当前样本变少了"，逐 ID 覆盖也查不出来。
    """
    from utils.batch import batch_paths

    paths = batch_paths(batch, generation)
    metadata_path = metadata_path or os.path.join(records_dir, "metadata.json")
    manifest_path = manifest_path or paths["variants"]
    problems: List[Problem] = []
    if not os.path.exists(metadata_path):
        problems.append(Problem("metadata_missing",
                                f"批次元数据不存在：{_rel(metadata_path)}"))
        return None, problems
    try:
        with open(metadata_path, encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, ValueError) as exc:
        problems.append(Problem("metadata_unreadable",
                                f"批次元数据无法解析：{_rel(metadata_path)}（{exc}）"))
        return None, problems
    if metadata.get("generation") != generation or metadata.get("batch") != batch:
        problems.append(Problem(
            "metadata_wrong_batch",
            f"批次元数据属于 {metadata.get('generation')}/{metadata.get('batch')}，"
            f"不是 {generation}/{batch}：{_rel(metadata_path)}"))
        return None, problems
    policy = metadata.get("policy")
    if not policy:
        problems.append(Problem("metadata_no_policy",
                                f"批次元数据没有 policy，无法还原 trajectory_id："
                                f"{_rel(metadata_path)}"))
        return None, problems
    if not os.path.exists(manifest_path):
        problems.append(Problem("manifest_missing",
                                f"变体清单不存在，无法核对期望样本集：{_rel(manifest_path)}"))
        return None, problems
    try:
        with open(manifest_path, encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError) as exc:
        problems.append(Problem("manifest_unreadable",
                                f"变体清单无法解析：{_rel(manifest_path)}（{exc}）"))
        return None, problems
    variants = payload.get("variants")
    if not isinstance(variants, dict):
        problems.append(Problem("manifest_unreadable",
                                f"变体清单没有 variants 字典：{_rel(manifest_path)}"))
        return None, problems
    ids = set()
    for entry in variants.values():
        variant_id = entry.get("variant_id")
        if not variant_id:
            problems.append(Problem("manifest_unreadable",
                                    f"变体清单有条目缺 variant_id：{_rel(manifest_path)}"))
            continue
        ids.add(f"{policy}__{variant_id}")
    return ids, problems


# ---------------------------------------------------------------------------
# 核对
# ---------------------------------------------------------------------------

def verify_review_coverage(generation: str, batch: str, *,
                           records_dir: Optional[str] = None,
                           ledger_path: Optional[str] = None,
                           manifest_path: Optional[str] = None,
                           metadata_path: Optional[str] = None,
                           expected_ids: Optional[Iterable[str]] = None) -> dict:
    """
    逐样本核对。返回一份结构化结果；`problems` / `missing` / `not_pass` /
    `fingerprint_mismatch` 任一非空即不能放行。
    """
    from utils.batch import batch_paths
    from utils.review_store import ledger_path as default_ledger

    records_dir = records_dir or batch_paths(batch, generation)["trajectories"]
    ledger_path = ledger_path or default_ledger(generation, batch)

    if expected_ids is None:
        expected, expected_problems = expected_ids_from_batch(
            generation, batch, records_dir,
            manifest_path=manifest_path, metadata_path=metadata_path)
    else:
        expected, expected_problems = set(expected_ids), []

    records, record_problems = load_records_strict(records_dir)
    traces, fingerprints, trace_problems = load_traces(records)
    decisions, ledger_problems = load_ledger_strict(generation, batch, ledger_path)
    current = latest(decisions)

    missing_records = sorted(expected - set(records)) if expected is not None else []
    unexpected_records = sorted(set(records) - expected) if expected is not None else []

    missing_decisions = sorted(set(records) - set(current))
    extra_decisions = sorted(set(current) - set(records))

    not_pass: Dict[str, List[str]] = {}
    invalid: List[Tuple[str, List[str]]] = []
    mismatched: List[Tuple[str, str, str]] = []
    for trajectory_id in sorted(set(records) & set(current)):
        decision = current[trajectory_id]
        validation = decision.validate()
        if validation:
            invalid.append((trajectory_id, validation))
            continue
        if decision.decision != "pass":
            not_pass.setdefault(decision.decision, []).append(trajectory_id)
            continue
        # 现场重算的指纹才算数；台账里的 stale 字段不作为依据。
        if trajectory_id in fingerprints and \
                decision.fingerprint != fingerprints[trajectory_id]:
            mismatched.append((trajectory_id, decision.fingerprint,
                               fingerprints[trajectory_id]))

    problems = (expected_problems + record_problems + trace_problems + ledger_problems)
    # 多出来的决定不抵扣覆盖，也不单独构成拒绝（台账是追加式的，历史 ID 可能
    # 仍留在文件里）；它必须被报告，但当前每个样本都要有自己的有效通过。
    ok = not (problems or missing_records or unexpected_records
              or missing_decisions or not_pass or invalid or mismatched)
    return {
        "generation": generation, "batch": batch,
        "records_dir": records_dir, "ledger_path": ledger_path,
        "expected": len(expected) if expected is not None else None,
        "current": len(records),
        "records": records,
        "fingerprints": fingerprints,
        "decisions": decisions,
        "problems": problems,
        "missing_records": missing_records,
        "unexpected_records": unexpected_records,
        "missing_decisions": missing_decisions,
        "extra_decisions": extra_decisions,
        "not_pass": not_pass,
        "invalid": invalid,
        "fingerprint_mismatch": mismatched,
        "ok": ok,
    }


def format_problems(result: dict, *, limit: int = 12) -> List[str]:
    """把核对结果变成可读的拒绝理由（具体到 ID）。"""
    lines: List[str] = []
    for problem in result["problems"]:
        lines.append(problem.render())
    if result["missing_records"]:
        lines.append("记录缺失（期望样本集里有、目录里没有）："
                     + "、".join(result["missing_records"][:limit])
                     + ("…" if len(result["missing_records"]) > limit else ""))
    if result["unexpected_records"]:
        lines.append("记录多出（不在期望样本集里）："
                     + "、".join(result["unexpected_records"][:limit])
                     + ("…" if len(result["unexpected_records"]) > limit else ""))
    if result["missing_decisions"]:
        lines.append("缺审（当前样本没有审核决定）："
                     + "、".join(result["missing_decisions"][:limit])
                     + ("…" if len(result["missing_decisions"]) > limit else ""))
    for decision in ("revise", "reject"):
        ids = result["not_pass"].get(decision) or []
        if ids:
            lines.append(f"未通过（{decision}）："
                         + "、".join(ids[:limit])
                         + ("…" if len(ids) > limit else ""))
    if result["invalid"]:
        shown = [f"{tid}（{'；'.join(reasons)}）"
                 for tid, reasons in result["invalid"][:limit]]
        lines.append("决定字段不合法：" + "、".join(shown)
                     + ("…" if len(result["invalid"]) > limit else ""))
    if result["fingerprint_mismatch"]:
        shown = [f"{tid}（台账 {saved}，当前 {current}）"
                 for tid, saved, current in result["fingerprint_mismatch"][:limit]]
        lines.append("审核指纹与当前内容不一致：" + "、".join(shown)
                     + ("…" if len(result["fingerprint_mismatch"]) > limit else ""))
    if result["extra_decisions"]:
        lines.append("台账里含当前批次不存在的 ID（不计入覆盖，不自动忽略）："
                     + "、".join(result["extra_decisions"][:limit])
                     + ("…" if len(result["extra_decisions"]) > limit else ""))
    return lines


def assert_review_accepted(generation: str, batch: str, *,
                           records_dir: Optional[str] = None,
                           ledger_path: Optional[str] = None,
                           manifest_path: Optional[str] = None,
                           metadata_path: Optional[str] = None,
                           expected_ids: Optional[Iterable[str]] = None) -> dict:
    """核对失败就 SystemExit，消息里逐项列出 ID；成功返回核对结果。"""
    result = verify_review_coverage(
        generation, batch, records_dir=records_dir, ledger_path=ledger_path,
        manifest_path=manifest_path, metadata_path=metadata_path,
        expected_ids=expected_ids)
    if not result["ok"]:
        lines = format_problems(result)
        raise SystemExit(
            f"外部 Agent 评测被拒绝：{generation}/{batch} 的逐样本验收未通过"
            f"（当前 {result['current']} 条记录，台账 {len(result['decisions'])} 行）。\n  "
            + "\n  ".join(lines)
            + "\n门禁不会自动改决定、也不会把样本改成通过；修复后重新审核/导入。")
    return result
