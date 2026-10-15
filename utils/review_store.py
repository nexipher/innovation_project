"""
g8 验收审核的存储层（阶段 4.6）。

**为什么这些规则在这里而不是页面里**：页面里的 JS 只能被浏览器验证，而"决定属于
哪一批、哪一条、哪一版内容"是**身份**问题——判错一次，别人几小时的审核就串到另一
批数据上，而且看不出来。所以指纹、合并、失效、导入校验都写成 Python，测试钉死；
页面里的 JS 只是同一套规则的薄客户端（草稿），写回仓库一律走
`scripts/g8_review_import.py`。

三条硬规则：

1. **身份＝(代次, 批次, 样本内容指纹)**。内容指纹对**页面展示的事实**取：判定、
   停止原因、计数器、引用检查、逐轮原文。重渲染而内容不变 → 指纹不变 → 决定保留；
   内容变了 → 旧决定标**已过期**，不自动继承、也不静默沿用。
2. **浏览器保存 ≠ 已写回仓库**。localStorage 只是草稿；仓库里的台账是
   `sft_data/<gen>/review/<batch>/dispositions.jsonl`，只能由导入命令追加，且要过
   下面同一套校验。
3. **验收不是训练**。这里没有、也不接受任何"可训练"字段：`通过` 只表示本次行为
   验收通过。训练准入是另一条流水线（且明确不接旧准入规则）。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = "g8_review_decisions/1"

#: 决定取值。`pass` = 本次审核通过（**不是**"可训练"）。
DECISIONS = ("pass", "revise", "reject")
#: 需要写原因的决定
NEEDS_REASON = ("revise", "reject")
#: 问题分类（revise / reject 用；可空，但决定为 revise/reject 时必须有备注文字）
CATEGORIES = (
    "visual_grounding",       # 视觉观察不具体/编造看不见的痕迹
    "evidence_citation",      # 引用不忠实、引用未投递字段、方向读反
    "tool_use",               # 该取第二意见没取、重复请求、缺席处理不当
    "protocol",               # 格式/标签/动作数量违规
    "stopping",               # 停止原因可疑（过早结案/该继续却停）
    "logging",                # 记录缺失或与页面不一致
    "other",
)

#: 内容指纹覆盖的记录字段：**页面展示、且审核者据此下判断**的那些事实。
FINGERPRINT_RECORD_FIELDS = (
    "trajectory_id", "variant_id", "source_id", "treatment", "container", "generator",
    "split", "ground_truth", "final_verdict", "model_candidate", "verdict_status",
    "confidence", "halting_reason", "policy_reasons", "report", "counters",
    "evidence_check", "tools_served", "record_schema", "image_path",
)
#: 内容指纹覆盖的 trace metadata 字段（审计面板展示的那些）。
FINGERPRINT_METADATA_FIELDS = (
    "image_size", "expert_call_count", "model_turn_count", "unique_evidence_count",
    "suppressed_duplicate_count", "skipped_duplicate_calls", "repeated_action_turns",
    "invalid_action_turns", "format_failure_turns", "unavailable_tool_calls",
    "failed_tool_calls", "verdict_status", "halting_reason", "policy_reasons",
    "model_abstained", "protocol_violations", "total_steps", "weighted_cost",
)


def sample_fingerprint(record: dict, trace: Optional[dict] = None) -> str:
    """
    一条样本**内容**的指纹：页面展示什么，就对什么取。

    刻意**不含**生成时间、文件路径以外的运行元数据、也不含页面本身的版本：改页面
    排版不该让谁的决定失效，而实测内容变了（换了图、改了判定、多了/少了一轮）
    必须让它失效。
    """
    trace = trace or {}
    metadata = trace.get("metadata") or {}
    payload = {
        "record": {name: record.get(name) for name in FINGERPRINT_RECORD_FIELDS},
        "metadata": {name: metadata.get(name) for name in FINGERPRINT_METADATA_FIELDS},
        "turns": [{"from": turn.get("from"), "value": turn.get("value"),
                   "image_paths": turn.get("image_paths")}
                  for turn in (trace.get("conversations") or [])],
        "evidence": record.get("evidence") or [],
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


@dataclass
class Decision:
    """一条决定。`stale` 由指纹比对得出，不由导入方声明。"""

    trajectory_id: str
    fingerprint: str
    decision: str = "pass"
    reviewer: str = ""
    category: Optional[str] = None
    note: str = ""
    decided_at: str = ""
    stale: bool = False
    stale_reason: str = ""

    def validate(self) -> List[str]:
        problems: List[str] = []
        if self.decision not in DECISIONS:
            problems.append(f"未知决定 {self.decision!r}（只接受 {DECISIONS}）")
        if not self.reviewer.strip():
            problems.append("审核人不能为空")
        if not self.fingerprint:
            problems.append("缺内容指纹")
        if self.decision in NEEDS_REASON and not (self.note or "").strip():
            problems.append(f"{self.decision} 必须填写原因")
        if self.category and self.category not in CATEGORIES:
            problems.append(f"未知问题分类 {self.category!r}")
        return problems

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint, "decision": self.decision,
            "reviewer": self.reviewer, "category": self.category, "note": self.note,
            "decided_at": self.decided_at, "stale": self.stale,
            "stale_reason": self.stale_reason,
        }


def export_payload(generation: str, batch: str, reviewer: str,
                   decisions: Dict[str, Decision]) -> dict:
    """导出：代次、批次、审核人、逐条决定（含指纹）。"""
    return {
        "schema": SCHEMA,
        "generation": generation,
        "batch": batch,
        "reviewer": reviewer,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        # 导出即声明：验收决定，不是"可训练"标记
        "scope": "acceptance_only",
        "decisions": {key: value.to_dict() for key, value in sorted(decisions.items())},
    }


def check_import(payload: dict, generation: str, batch: str,
                 fingerprints: Dict[str, str]) -> Tuple[List[Decision], List[dict]]:
    """
    校验一份导入：**拒绝**跨代次/跨批次/未知样本，**标过期**指纹不符的决定。

    返回 (接受的决定, 拒绝清单)。拒绝的理由逐条写清楚——静默丢弃决定比拒绝更糟。
    """
    accepted: List[Decision] = []
    rejected: List[dict] = []

    if payload.get("schema") != SCHEMA:
        return [], [{"trajectory_id": "*", "reason": f"schema 不是 {SCHEMA}"}]
    if payload.get("generation") != generation or payload.get("batch") != batch:
        return [], [{"trajectory_id": "*",
                     "reason": (f"跨代次/跨批次：文件是 "
                                f"{payload.get('generation')}/{payload.get('batch')}，"
                                f"当前是 {generation}/{batch}")}]
    if payload.get("scope") not in (None, "acceptance_only"):
        return [], [{"trajectory_id": "*", "reason": f"未知用途 {payload.get('scope')!r}"}]

    for trajectory_id, raw in (payload.get("decisions") or {}).items():
        if trajectory_id not in fingerprints:
            rejected.append({"trajectory_id": trajectory_id,
                             "reason": "本批次没有这条样本"})
            continue
        decision = Decision(
            trajectory_id=trajectory_id,
            fingerprint=str(raw.get("fingerprint") or ""),
            decision=str(raw.get("decision") or ""),
            reviewer=str(raw.get("reviewer") or payload.get("reviewer") or ""),
            category=raw.get("category"),
            note=str(raw.get("note") or ""),
            decided_at=str(raw.get("decided_at") or ""),
        )
        problems = decision.validate()
        if problems:
            rejected.append({"trajectory_id": trajectory_id,
                             "reason": "；".join(problems)})
            continue
        current = fingerprints[trajectory_id]
        if decision.fingerprint != current:
            decision.stale = True
            decision.stale_reason = (f"内容已变：{decision.fingerprint} → {current}"
                                     "（不自动继承，需重审）")
        accepted.append(decision)
    return accepted, rejected


def merge_decisions(existing: Iterable[Decision],
                    incoming: Iterable[Decision]) -> List[Decision]:
    """
    台账合并：同一条**已决定**的样本，仓库里已有的记录为历史，新的追加在后面。

    审核会改主意，也会有多位审核人；所以台账是追加式的流水，不是覆盖式的最新值
    ——"谁在什么时候把哪一条从 revise 改成 pass"本身就是审计信息。
    """
    merged = list(existing)
    for decision in incoming:
        merged.append(decision)
    return merged


def latest(decisions: Sequence[Decision]) -> Dict[str, Decision]:
    """每条样本的**最新**决定（按 time/顺序），供进度统计与页面显示。"""
    out: Dict[str, Decision] = {}
    for decision in decisions:
        out[decision.trajectory_id] = decision
    return out


def progress(decisions: Sequence[Decision], total: int) -> dict:
    """进度统计：已决定/未决定/有效的各决定数/过期数/分类计数。"""
    current = latest(decisions)
    valid = {key: value for key, value in current.items() if not value.stale}
    stale = {key: value for key, value in current.items() if value.stale}
    by_decision = {name: sum(1 for value in valid.values() if value.decision == name)
                   for name in DECISIONS}
    by_category: Dict[str, int] = {}
    for value in valid.values():
        if value.category:
            by_category[value.category] = by_category.get(value.category, 0) + 1
    return {
        "total": total,
        "decided": len(valid),
        "undecided": max(0, total - len(valid)),
        "stale": len(stale),
        "by_decision": by_decision,
        "by_category": by_category,
    }


# ---------------------------------------------------------------------------
# 仓库侧台账（追加式 jsonl）
# ---------------------------------------------------------------------------

def ledger_path(generation: str, batch: str) -> str:
    from config import PROJECT_ROOT

    return os.path.join(PROJECT_ROOT, "sft_data", generation, "review", batch,
                        "dispositions.jsonl")


def append_ledger(generation: str, batch: str, decisions: Iterable[Decision],
                  path: Optional[str] = None) -> str:
    """把决定**追加**写进仓库台账。返回写入的文件路径。"""
    target = path or ledger_path(generation, batch)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "a", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps({
                "generation": generation, "batch": batch,
                "trajectory_id": decision.trajectory_id,
                "stale": decision.stale, "stale_reason": decision.stale_reason,
                **decision.to_dict(),
                "imported_at": datetime.now().isoformat(timespec="seconds"),
            }, ensure_ascii=False) + "\n")
    return target


def load_ledger(generation: str, batch: str,
                path: Optional[str] = None) -> List[Decision]:
    """读回台账（追加式：同一样本可能有多条，`latest` 取最后一条）。"""
    target = path or ledger_path(generation, batch)
    if not os.path.exists(target):
        return []
    out: List[Decision] = []
    with open(target, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            out.append(Decision(
                trajectory_id=raw.get("trajectory_id", ""),
                fingerprint=str(raw.get("fingerprint") or ""),
                decision=raw.get("decision", ""),
                reviewer=raw.get("reviewer", ""),
                category=raw.get("category"),
                note=raw.get("note", ""),
                decided_at=raw.get("decided_at", ""),
                stale=bool(raw.get("stale")),
                stale_reason=raw.get("stale_reason", ""),
            ))
    return out
