"""Evidence-coverage checks for the two joint answers.

The rule is narrow and does not choose a label:

* a definite ``whole_image_generation`` needs a served PROBE measurement;
* a definite ``local_manipulation`` needs a served TruFor measurement;
* ``whole_image_generation = Uncertain`` / ``local_manipulation = Uncertain``
  are always allowed, so a single-expert conclusion (for example, TruFor-local
  evidence supporting an overall Fake while the whole-image question stays
  Uncertain) remains legal;
* cross-expert "mutually supports / cross-validates" wording is recorded when
  both measurements exist, unless the sentence negates it;
* a definite edit-process claim (for example "text was added later") is
  actionable wording, because an anomaly map is a response, not an edit
  history; a negated mention is left alone.

The controller may use the actionable issues for one targeted correction.  It
never rewrites the model's fields itself.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

MUTUAL_STRONG = re.compile(
    r"(?:相互验证|互相验证|相互印证|互相印证|双重验证|交叉验证|交叉印证|"
    r"共同支持|彼此印证|相互支持|同时高分|均高|"
    r"mutual(?:ly)? valid|cross.?validat|corroborat|high agreement|both high)",
    re.I)
NEGATION = re.compile(
    r"(?:不|无需|无须|并非|不是|未|没有|无|no|not|without|cannot|can't)\s*$",
    re.I)
PROCESS_CLAIM = re.compile(
    r"(?:后期添加|后期合成|后期编辑|添加了|被添加|被合成|抠图|合成到|"
    r"拼接到|移花接木|added later|pasted|composited)", re.I)

ACTIONABLE = ("whole_image_without_probe", "local_manipulation_without_trufor",
              "cross_task_mutual_validation", "edit_process_claim")


def _measured_sources(evidence_chain: Iterable[dict]) -> set:
    return {
        token.get("source")
        for token in evidence_chain or []
        if token.get("source") and token.get("status", "measured") != "failed"
    }


def _clause_prefix(text: str, start: int, window: int) -> str:
    """Text before a match, cut at the latest sentence boundary."""
    prefix = text[max(0, start - window):start]
    cut = max((prefix.rfind(ch) for ch in "。；;！？\n"), default=-1)
    return prefix[cut + 1:] if cut >= 0 else prefix


def _negated(text: str, match) -> bool:
    prefix = _clause_prefix(text, match.start(), 24)
    if NEGATION.search(prefix):
        return True
    # "两者结果一致，但不构成相互验证" negates the phrase in the same clause
    # without putting the negator immediately before it.  Do not treat a bare
    # "不" in "不同维度" as negation unless it is one of these phrases.
    return bool(re.search(
        r"(?:不构成|不能|不应|不要|不可|无法|并非|不是|没有|无需|无须)",
        prefix, re.I))


def coverage_issues(candidate: dict, text: str, evidence_chain: Iterable[dict],
                    served: Iterable[str]) -> List[dict]:
    """Return unsupported definite fields and cross-task wording issues."""
    candidate = candidate or {}
    measured = _measured_sources(evidence_chain)
    served = set(served or [])
    issues: List[dict] = []
    whole = candidate.get("whole_image_generation")
    local = candidate.get("local_manipulation")
    if (whole in ("Real", "AI-generated")
            and "probe_dino_expert" in served
            and "probe_dino_expert" not in measured):
        issues.append({
            "code": "whole_image_without_probe",
            "severity": "defect",
            "field": "whole_image_generation",
            "value": whole,
            "message": ("whole_image_generation is definite without a PROBE "
                        "measurement; the whole-image question is unmeasured"),
        })
    if (local in ("Tampered", "Not tampered")
            and "trufor_expert" in served
            and "trufor_expert" not in measured):
        issues.append({
            "code": "local_manipulation_without_trufor",
            "severity": "defect",
            "field": "local_manipulation",
            "value": local,
            "message": ("local_manipulation is definite without a TruFor "
                        "measurement; the local question is unmeasured"),
        })
    if ("probe_dino_expert" in measured and "trufor_expert" in measured
            and text):
        for match in MUTUAL_STRONG.finditer(text):
            if _negated(text, match):
                continue
            issues.append({
                "code": "cross_task_mutual_validation",
                "severity": "wording",
                "phrase": match.group(0),
                "message": ("PROBE and TruFor answer different questions; do not "
                            "call their results mutual validation/support"),
            })
            break
    if text and "trufor_expert" in measured and local == "Tampered":
        match = PROCESS_CLAIM.search(text)
        if match:
            prefix = _clause_prefix(text, match.start(), 24)
            negated = bool(re.search(
                r"(?:不能|无法|不应|不要|不可|并未|没有|并非|不是|不|"
                r"未发现|未见|未显示|未确认|不能确认|无法确认)",
                prefix, re.I))
            if not negated:
                issues.append({
                    "code": "edit_process_claim",
                    "severity": "wording",
                    "phrase": match.group(0),
                    "message": ("an anomaly map is a response, not an edit "
                                "history; do not assert which edit was performed"),
                })
    return issues


def actionable(issues: Iterable[dict]) -> List[dict]:
    return [issue for issue in issues or []
            if issue.get("code") in ACTIONABLE]


def coverage_note(issues: Iterable[dict]) -> str:
    """One targeted correction; it never supplies a label."""
    issues = list(issues or [])
    lines = [
        "[System: 证据覆盖检查：以下分项或措辞没有得到本会话的测量支持。"
        "请保留由证据支持的总体结论，只修正这些分项/措辞后重新输出 <verdict>；"
        "也可以先补测缺失的专家（只输出一个 <call_x>），不要同时给调用和 verdict；"
        "不要伪造专家分数。"
    ]
    for issue in issues:
        code = issue.get("code")
        if code == "whole_image_without_probe":
            lines.append(
                f"- whole_image_generation={issue.get('value')}，但本会话没有 "
                "PROBE 的有效测量。局部证据可以支持总体 Fake，但整图生成分项"
                "应写 Uncertain，或明确说明非 PROBE 的依据边界；不要把 TruFor "
                "的局部高分改写成整图生成结论。若因此把整图改为 Uncertain，"
                "local_manipulation 的 Not applicable 也必须改为 Uncertain"
                "（Not applicable 只在整图 AI-generated 时成立）。")
        elif code == "local_manipulation_without_trufor":
            lines.append(
                f"- local_manipulation={issue.get('value')}，但本会话没有 "
                "TruFor 的有效测量。局部问题未测时应写 Uncertain；若整图已判 "
                "AI-generated 且局部不适用，才可写 Not applicable。")
        elif code == "cross_task_mutual_validation":
            lines.append(
                f"- “{issue.get('phrase')}”把 PROBE 和 TruFor 写成相互验证/"
                "相互支持。两者测量不同问题，只能分别说明各自贡献；若同一句"
                "已经否定该说法，可保留否定表述。")
        elif code == "edit_process_claim":
            lines.append(
                f"- “{issue.get('phrase')}”从异常图断言了具体编辑过程。异常图"
                "只是响应候选，不是编辑历史；请改为描述响应位置/强度及其局限。")
    lines.append(
        "如果某维确实无法测量，如实写 Uncertain 并说明是工具失败、未调用还是"
        "没有适用工具；不要为了保持原结论而编造测量。]")
    return "\n".join(lines)
