"""
final_v2 schema and validator (plan.md §4.11 G4-d).

A training sample in `final_v2` is not a raw trace: it is a structured answer
built from an admitted trajectory, and every claim in it must be checkable
against the record it came from.

The answer has four sections:

    <observation>         facts verifiable in the image itself
    <forensic_evidence>   evidence_id, expert, measurement scope, raw metric,
                          calibrated likelihood, applicability
    <reasoning>           support, counter-evidence, alternative explanations,
                          why an expert failed here, what remains uncertain
    <verdict>             the label and confidence the halting posterior gave

The validator is the point of the module.  It re-derives each check from the
record rather than trusting the text, and it covers the failures this project
has actually produced: a source in two splits, a referenced evidence id that
does not exist, a direction the calibration contradicts, an expert used where
it was measured not to separate, a verdict that disagrees with its own
posterior, confidence without admissible evidence, and — the one that cost two
runs — the container format written up as a reason for the label.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Iterable, List, Optional, Tuple

from config import PROJECT_ROOT

SECTIONS = ("observation", "forensic_evidence", "reasoning", "verdict")
SCHEMA_VERSION = "final_v2"
# The trajectory schema that carries the evidence ledger's fields.  A record
# declaring it must have them; records without the declaration are legacy.
RECORD_SCHEMA = "g7_r4"

# Confidence at which a label has to be backed by admissible evidence.
CONFIDENT_THRESHOLD = 0.8
# Experts retired in G2-e: never admissible, in any sample.
RETIRED_EXPERTS = ("frequency_expert", "ela_expert")

# Container talk is only a defect when it is used as a reason for the label;
# a sentence that explicitly denies the inference is exactly what we want the
# model to write.
CONTAINER_WORDS = ("png", "jpeg", "jpg", "容器", "container", "格式")
DIRECTION_WORDS = ("伪造", "篡改", "ai 生成", "ai生成", "真实", "fake", "real",
                   "generated", "forgery", "authentic")
# A container only becomes a reason when it is *used* as one: the sentence has
# to draw an inference from it, which a bare mention does not.
CAUSAL_CONNECTIVES = ("因为", "由于", "所以", "因此", "据此", "说明", "意味着", "表明",
                      "indicates", "because", "therefore", "suggests", "proves")
# ... and this is what a correct answer sounds like: it names the trap and
# denies it, which must not be mistaken for falling into it.
CONTAINER_DENIALS = ("不代表", "不能作为", "不能据此", "与伪造无关", "not a reason",
                     "does not indicate", "no bearing", "不构成", "无关", "不能说明",
                     "不应", "不得", "不能", "而非", "而不是", "不意味着")

_SECTION_PATTERN = re.compile(r"<(observation|forensic_evidence|reasoning|verdict)>(.*?)</\1>", re.S)
# The model's own reasoning block, as it wrote it (the rendered answer uses it
# verbatim rather than replacing it with a template).
_REASONING_BLOCK = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.S | re.I)
# The verdict JSON the model wrote, read back for the one field the rendered
# answer used to drop (`report`) — see `adopted_report`.
_VERDICT_JSON = re.compile(r"<verdict>\s*(\{.*?\})\s*</verdict>", re.S | re.I)
_SENTENCE_SPLIT = re.compile(r"[。.;；\n]")

# Grounding (2026-10-11): the assistant may cite an expert's *results* only if
# that expert ran.  A call-shaped mention is an action — the controller decides
# whether it executes — but "the frequency expert's analysis shows ..." is a
# claim about a measurement, and 15 samples in the review contained one for an
# expert that was never invoked.
TOOL_EXPERTS = {"noise": "noise_expert", "jpeg": "jpeg_expert",
                "freq": "frequency_expert_v2", "frequency": "frequency_expert_v2",
                # G7: PROBE leads the toolkit now, so its token is cited by
                # name in the prose and must be recognised as a known expert.
                "probe": "probe_dino_expert", "probe_dino": "probe_dino_expert"}
_CALL_TAG = re.compile(r"<call_(\w+)>\s*(\[)?")
_PLANNING_BLOCK = re.compile(r"<planning>.*?</planning>", re.S)

# A global measurement described as a local one: the noise expert's own
# "Localised noise variance ..." is the canonical case.
_LOCAL_MEASUREMENT = re.compile(
    r"(localis|local(?:ized)?\s+(?:variance|noise|residual)|局部|该区域|区域内)"
    r"[^。.;\n]{0,24}(方差|噪声|残差|纹理|频谱|variance|noise|residual)", re.I)
# Observation lines that describe the file rather than the picture.
_METADATA_LINE = re.compile(r"^[-*\s]*(container|quality|resolution|格式|分辨率)\b", re.I)


def conclusion_gate(record: dict) -> Optional[str]:
    """
    Why this trajectory may not become training data, or None (§4.17 R4).

    Four things are separate and must stay separate: the evidence check, the
    format of the adopted answer, whether the conclusion is faithful, and
    whether the sample is trainable.  This function is the last one — the hard
    gate that keeps a session whose fabrication survived its single correction
    out of the training set (and out of admission scoring, which calls it too).

    Legacy trajectories (generated before the ledger existed) have no
    `evidence_check` key at all; they are not retroactively re-judged here —
    the audit of those trajectories is a separate, recorded exercise.
    """
    if record.get("record_schema") == RECORD_SCHEMA:
        # A batch that declares the schema must carry its fields: a missing
        # one is a defect in that batch, not a legacy trajectory, and must
        # never be waved through by the compatibility path.
        for field_name in ("evidence_check", "conclusion"):
            if record.get(field_name) is None:
                return f"{field_name}_missing_for_schema"
    if "evidence_check" not in record:
        return None
    check = record.get("evidence_check") or {}
    status = check.get("status")
    if status == "failed":
        return "evidence_check_failed"
    if status not in ("pass", "pass_after_correction"):
        return f"evidence_check_{status or 'missing'}"
    conclusion = record.get("conclusion")
    if conclusion is None:
        return None                     # written by a pre-R4 generator
    if conclusion.get("verdict_parse") != "ok":
        return "conclusion_without_parseable_verdict"
    if conclusion.get("calls") and not conclusion.get("calls_executed"):
        return "conclusion_contains_unexecuted_calls"
    return None


def _tool_mentions(text: str):
    """(expert, is_call_shaped) for every tool the assistant refers to."""
    mentions = []
    planning = " ".join(match.group(0) for match in _PLANNING_BLOCK.finditer(text))
    for match in _CALL_TAG.finditer(text):
        expert = TOOL_EXPERTS.get(match.group(1).lower())
        if expert is None:
            continue
        followed_by_region = bool(match.group(2))
        # Inside a planning block, or followed by coordinates: the turn is
        # proposing a call, not reading results back.
        mentions.append((expert, followed_by_region or match.group(0) in planning))
    for expert in ("noise_expert", "jpeg_expert", "frequency_expert_v2",
                   "probe_dino_expert"):
        if expert in text:
            mentions.append((expert, False))
    return mentions


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_evidence_block(token: dict) -> str:
    """One evidence entry: identity, scope, measurement, calibration, caveats."""
    likelihood = token.get("calibrated_likelihood") or {}
    lines = [
        f"- evidence_id: {token.get('evidence_id')}",
        f"  expert: {token.get('source')}",
        f"  measurement_scope: {token.get('measurement_scope', 'region')}",
        f"  raw_metric: {token.get('raw_metric')}",
        f"  region (diagnostic): {token.get('region')}",
    ]
    if likelihood:
        lines.append("  calibrated_likelihood: "
                     f"Real={likelihood.get('Real')}, Fake={likelihood.get('Fake')}")
    if token.get("applicability"):
        lines.append(f"  applicability: {token['applicability']}")
    if token.get("applicability_conditions"):
        lines.append(f"  conditions: {token['applicability_conditions']}")
    return "\n".join(lines)


def _observed_size(record: dict):
    """
    The size of the image the model actually saw, and where the number came from.

    The observation section printed `resolution: None` for every sample of the
    v3 batch: the field it read is filled by one caller and not another, while
    the number was there all along in the trace (`image_size`).  A conversion
    that says nothing is better than one that says "None" — but the size is a
    fact about the input, so it is carried, with its provenance.
    """
    if record.get("resolution"):
        return record["resolution"], "源图分辨率（split）"
    metadata = ((record.get("_trace") or {}).get("metadata") or {})
    if metadata.get("image_size"):
        return metadata["image_size"], "输入图像尺寸（模型实际所见）"
    token = next(iter(record.get("evidence") or []), {})
    if token.get("image_height") and token.get("image_width"):
        return [token["image_height"], token["image_width"]], "证据 token 记录"
    return None, "未记录"


def render_observation(record: dict, observation_text: str = "") -> str:
    """
    What can be checked in the image itself, plus how it was delivered.

    The model's own words come first and are never written by this function:
    if the session did not describe the image, the section says less, not
    more.  (Telling the model to describe the scene is the *protocol's* job —
    §4.17 next-batch prompt — not something to backfill here.)
    """
    size, provenance = _observed_size(record)
    facts = [
        f"container: {record.get('container')}, quality: {record.get('quality', record.get('treatment'))}",
        f"resolution: {size if size else '未记录'}（{provenance}）",
    ]
    body = "\n".join(f"- {fact}" for fact in facts)
    if observation_text:
        body = observation_text.strip() + "\n" + body
    return body


def render_reasoning(record: dict, model_reasoning: str = "") -> str:
    """
    The model's own argument, then the evidence bookkeeping.

    The section used to be built entirely from token metadata: a clean,
    well-formed paragraph that could sit on top of an answer the model never
    reasoned that way through — the audit found answers whose *rendered*
    reasoning was better than the session's.  The model's own words come
    first and verbatim; the structured blocks below are what the pipeline
    knows about the evidence (support, counter-explanations, failure
    conditions, the posterior), labelled as such.  Nothing is smoothed over:
    a session whose argument is defective is refused upstream, not rewritten
    here.
    """
    tokens = record.get("evidence") or []
    head = ""
    if model_reasoning.strip():
        head = "模型自身论证（原文，未经改写）：\n" + model_reasoning.strip() + "\n\n"
    supporting = [t for t in tokens if t.get("support") in ("Real", "AI-generated")]
    lines = []
    if supporting:
        lines.append("支持的证据：")
        lines += [f"- {t.get('source')}: {t.get('interpretation_text', '')}".rstrip()
                  for t in supporting]
    counter = [t.get("counter_explanation") for t in tokens if t.get("counter_explanation")]
    if counter:
        lines.append("反证与替代解释：")
        lines += [f"- {text}" for text in counter[:3]]
    caveats = [t.get("applicability_conditions") for t in tokens if t.get("applicability_conditions")]
    if caveats:
        lines.append("专家失效条件：")
        lines += [f"- {text}" for text in caveats[:3]]
    if head:
        lines.insert(0, "证据簿记（由 token 生成，不是对模型论证的改写）：")
    if tokens:
        lines.append(
            f"剩余不确定性：后验 P(Fake)={record.get('posterior')}，"
            f"冲突度={record.get('conflict_score')}，停止原因={'/'.join(record.get('policy_reasons') or [])}"
        )
    else:
        # No tools ran, so there is no posterior to report: the 0.5 in the
        # record is the prior the session started from, and writing it as
        # "后验 P(Fake)=0.5" claims a measurement that never happened.
        lines.append(
            "剩余不确定性：本轨迹未获得工具证据（无工具协议），结论来自模型的直接视觉判断，"
            "没有测量后验可引用。"
        )
    return head + "\n".join(lines)


def render_answer(record: dict, observation_text: str = "",
                  model_reasoning: str = "") -> str:
    """
    The structured assistant turn final_v2 trains on.

    The verdict block says where the label came from, and the answer differs
    by policy because the policies do.  Under v3 the label is the model's own
    conclusion; the posterior is the *reference* the session recorded beside
    it.  Writing `basis: evidence_posterior` there (which is what the
    converter did) would tell the next model that a rule decided the label —
    the exact confusion the repair removed.
    """
    tokens = record.get("evidence") or []
    evidence_blocks = "\n".join(render_evidence_block(t) for t in tokens)
    model_led = record.get("policy_version") == "v3"
    if not tokens:
        # A tool-free conclusion rests on the model's perception; the 0.5 in
        # the record is the prior, and calling it a posterior claims a
        # measurement that never happened.
        basis = "model_perception"
    else:
        basis = "model_judgement" if model_led else "evidence_posterior"
    verdict = {
        "verdict": record.get("final_verdict"),
        "verdict_status": record.get("verdict_status") or (
            "model_concluded" if model_led else "policy_label"),
        "confidence": record.get("confidence"),
        "basis": basis,
    }
    # 模型自己写的那段 `report` 原文带上：推理提示词要求它出现在判决 JSON 里，
    # 而转换曾经把它整段丢掉——训练出来的模型于是被教成写一个不含 report 的判决，
    # 与它手上的说明相反。没有 trace（或那一轮没写）时不含此字段，绝不代写。
    model_report = adopted_report(record.get("_trace"))
    if model_report:
        verdict["report"] = model_report
    if tokens:
        # The reference lives apart from the label under v3.
        verdict["reference_posterior"] = record.get("posterior")
        if model_led:
            reference = record.get("reference") or {}
            verdict["reference_recommendation"] = reference.get("recommendation")
        else:
            verdict["posterior"] = record.get("posterior")
    return (
        f"<observation>\n{render_observation(record, observation_text)}\n</observation>\n\n"
        f"<forensic_evidence>\n{evidence_blocks or '- 本轨迹未调用工具'}\n</forensic_evidence>\n\n"
        f"<reasoning>\n{render_reasoning(record, model_reasoning)}\n</reasoning>\n\n"
        f"<verdict>\n{json.dumps(verdict, ensure_ascii=False)}\n</verdict>"
    )


def _is_evidence_turn(value: str) -> bool:
    """A delivery turn *is* the token JSON — a message that mentions one is not."""
    stripped = (value or "").lstrip()
    return stripped.startswith("{") and '"evidence_id"' in stripped[:400]


def _closing_index(trace_turns: List[dict]) -> int:
    """
    The assistant turn that ends the session.

    Normally the structured four-section answer; when a trace recorded only an
    older, unstructured conclusion, that last assistant turn is the one the
    rendered answer replaces.
    """
    structured = [index for index, turn in enumerate(trace_turns)
                  if turn.get("from") == "gpt"
                  and "<observation>" in (turn.get("value") or "")
                  and "<verdict>" in (turn.get("value") or "")]
    if structured:
        return structured[-1]
    assistant = [index for index, turn in enumerate(trace_turns)
                 if turn.get("from") == "gpt"]
    return assistant[-1] if assistant else -1


def adopted_reasoning(trace: Optional[dict]) -> str:
    """
    The model's own reasoning, from the turn the pipeline adopted.

    `_closing_index` names that turn; its <reasoning> block is what the model
    actually argued.  Empty when the trace predates the block or the block is
    empty — in which case the rendered section says so instead of dressing a
    template up as the model's reasoning.
    """
    trace_turns = (trace or {}).get("conversations", [])
    index = _closing_index(trace_turns)
    if index < 0:
        return ""
    block = _REASONING_BLOCK.search(trace_turns[index].get("value") or "")
    return block.group(1).strip() if block else ""


def adopted_report(trace: Optional[dict]) -> str:
    """
    The model's own `report`, from the same adopted turn the reasoning comes from.

    推理提示词要求判决 JSON 里带 `report`，而渲染出的答案里**没有这个字段**——训练
    出来的模型因此被教成"写一个不含 report 的判决"，与它自己的说明相反。这里把它
    原样取回来（同一个被采纳的轮次，绝不用别的轮次补齐）。
    """
    trace_turns = (trace or {}).get("conversations", [])
    index = _closing_index(trace_turns)
    if index < 0:
        return ""
    block = _VERDICT_JSON.search(trace_turns[index].get("value") or "")
    if not block:
        return ""
    try:
        parsed = json.loads(block.group(1))
    except ValueError:
        return ""
    report = parsed.get("report") if isinstance(parsed, dict) else None
    return str(report).strip() if isinstance(report, str) else ""


def render_conversation(record: dict, trace: Optional[dict] = None,
                        observation_text: str = "") -> List[dict]:
    """
    Rebuild the training conversation the way the session actually ran.

    Faithful replay (plan.md §4.11, 2026-10-11).  The pipeline injects a
    "[System: ...]" correction whenever it refuses a conclusion and the model
    revises.  Dropping those corrections while keeping both answers left two
    conclusions sitting next to each other with nothing to explain the change
    — 51 of 237 samples — which trains the model to reverse itself for no
    reason.  Every turn is kept, in order; only the closing answer is replaced
    by the rendered four-section answer, and everything before it is what the
    model actually said.

    A tool sample's answer cites evidence ids, so the conversation has to
    contain the turns that delivered them.  Those turns are rebuilt from the
    record's (rectified) tokens rather than copied from the trace, so the text
    the model trains on is the text the pipeline actually injects — in the
    shape it was actually delivered in (the slim pack on the g8 mainline, the
    whole token for full-delivery sessions), matched to its turn by
    `evidence_id` so a failed call cannot shift the alignment.  Turns after
    the closing answer are dropped: the sample's verdict is the one derived
    from that last answer.
    """
    from state_machine.evidence_tokenizer import EvidenceTokenizer

    evidence = record.get("evidence") or []
    trace_turns = (trace or {}).get("conversations", [])
    task_prompt = next((t["value"] for t in trace_turns
                        if t.get("from") == "user"), "<image>\n请分析这张图像的真实性。")

    # 交付轮按 **evidence_id** 对回 token，而不是按出现次序。阶段 4.4 起会话可以在
    # 一次执行失败后继续：失败记录也进证据链，但它**没有投递轮**。按次序对齐会把
    # 下一个 token 顶到上一轮的位置，训练对话里就会出现模型当时没拿到的测量。
    by_id = {token.get("evidence_id"): token for token in evidence
             if isinstance(token, dict) and token.get("evidence_id")}
    claimed: set = set()

    def _deliverable(token: dict) -> bool:
        """
        这个 token 当时**投递给模型了吗**。

        失败记录与 `model_delivered: "none"` 的记录都在证据链上，但模型什么都没
        拿到——它们没有投递轮，也不能顶替别人的位置。
        """
        return (token.get("status") != "failed"
                and token.get("model_delivered") != "none")

    def _claim(carried: dict) -> Optional[dict]:
        """
        这一段投递对应哪个 token。

        先认 `evidence_id`（真实 trace 里投递的就是链上那个 token，id 必然相同）。
        认不到时（旧 trace、或 id 在整流后变过）退回按次序取第一个**可投递**的
        token——跳过失败记录与从未投递的记录，它们没有投递轮，不该顶替别人的位置。
        """
        carried_id = (carried or {}).get("evidence_id")
        token = by_id.get(carried_id)
        if token is not None and carried_id not in claimed:
            if not _deliverable(token):
                return None              # 这一条当时没投递给模型
            claimed.add(carried_id)
            return token
        for candidate in evidence:
            if not isinstance(candidate, dict) or not _deliverable(candidate):
                continue
            candidate_id = candidate.get("evidence_id")
            if not candidate_id or candidate_id in claimed:
                continue
            claimed.add(candidate_id)
            return candidate
        return None

    closing = _closing_index(trace_turns)
    model_reasoning = adopted_reasoning(trace)
    turns: List[dict] = [{"from": "user", "value": task_prompt}]
    for index, turn in enumerate(trace_turns):
        if index == 0 and turn.get("from") == "user":
            continue                       # the task prompt is already in place
        value = (turn.get("value") or "").strip()
        if not value:
            continue
        if index == closing:
            turns.append({"from": "gpt", "value": render_answer(
                record, observation_text, model_reasoning)})
            break
        if turn.get("from") == "gpt":
            turns.append({"from": "gpt", "value": value})
            continue
        if _is_evidence_turn(value):
            token = _claim(_parse_evidence_turn(value))
            if token is None:
                # 记录里已经没有可交付的 token（例如记录说本轨迹没调工具）：
                # 这一段不属于这个样本，丢掉而不是照抄一个账上不存在的东西。
                continue
            # 按**当时投递的形状**重建：精简包就是精简包（阶段 4.3 起的主线），
            # 旧批次与完整投递仍是整个 token——训练与推理必须看到同一种字段。
            payload = (EvidenceTokenizer.model_visible(token)
                       if token.get("model_delivered") == "slim" else token)
            entry = {"from": "user", "value": EvidenceTokenizer.to_json(payload)}
            paths = [path for path in
                     ([token.get("diagnostic_region_image")]
                      + list(token.get("visual_artifacts") or [])) if path]
            if paths:
                # Inference attaches at most two images per turn; mirror it.
                entry["image_paths"] = paths[:2]
            turns.append(entry)
            continue
        turns.append({"from": "user", "value": value})   # question or correction
    else:
        # No closing turn in the trace: end the conversation with the answer.
        turns.append({"from": "gpt", "value": render_answer(
            record, observation_text, model_reasoning)})
    return turns


def audit_dialogue(record: dict) -> dict:
    """
    Every supervised assistant turn, judged **against what it could see then**.

    The runtime check covers the adopted conclusion; the training conversation
    keeps every earlier assistant turn as it was written, and a fabrication in
    one of those is supervised just as much.  The turn `render_answer`
    replaces is excluded — its text does not survive into the sample.

    Time matters.  The first version judged every turn with the *final*
    evidence chain "so a later value is still a legal quotation", which
    vindicates an early claim with evidence that had not arrived yet: a turn
    that invents the noise expert's result passes because noise is called two
    turns later.  Each turn is therefore judged with the ledger as of that
    turn — the tokens delivered *before* it — and the closing turn alone may
    quote whatever the session eventually collected.
    """
    from utils.evidence_ledger import Ledger

    trace = record.get("_trace") or {}
    trace_turns = trace.get("conversations", [])
    closing = _closing_index(trace_turns)
    if not trace_turns:
        return {"turns": 0, "defects": [], "candidates": [],
                "repeat_request_turns": []}

    served = record.get("tools_served") or []
    system_texts = [turn.get("value") or "" for turn in trace_turns
                    if (turn.get("from") or "") == "user"
                    and (turn.get("value") or "").startswith("[System:")]

    delivered: List[dict] = []
    defects: List[dict] = []
    candidates: List[dict] = []
    repeats: List[int] = []
    supervised = 0
    for index, turn in enumerate(trace_turns):
        role = (turn.get("from") or "")
        value = turn.get("value") or ""
        if role == "user":
            if _is_evidence_turn(value):
                token = _parse_evidence_turn(value)
                if token:
                    delivered.append(token)
            continue
        if role != "gpt":
            continue
        following = trace_turns[index + 1] if index + 1 < len(trace_turns) else {}
        if (following.get("from") or "") == "user" and "已跳过执行" in (
                following.get("value") or ""):
            repeats.append(index)
        if index == closing:
            continue
        supervised += 1
        # **As of this turn**: only what had been delivered when it was written.
        ledger = Ledger.from_session(delivered, served=served,
                                     system_texts=system_texts)
        for finding in ledger.findings(value):
            entry = {"turn": index, **finding}
            (defects if finding.get("severity", "defect") == "defect"
             else candidates).append(entry)
    return {"turns": supervised, "defects": defects, "candidates": candidates,
            "repeat_request_turns": repeats}


def _parse_evidence_turn(value: str) -> Optional[dict]:
    """The token a delivery turn carried (the trace stores what was injected)."""
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) and parsed.get("evidence_id") else None


def render_sample(record: dict, category: str, observation_text: str = "",
                  split_entry: Optional[dict] = None,
                  trace: Optional[dict] = None) -> dict:
    """Assemble a final_v2 sample from one admitted trajectory."""
    image_path = record.get("image_path") or record.get("variant_path", "")
    record = {**record, "_trace": trace}
    model_reasoning = adopted_reasoning(trace)
    dialogue = audit_dialogue(record)
    sample = {
        "id": f"f2_{record['trajectory_id']}",
        "image_path": image_path,
        "ground_truth": record["ground_truth"],
        "source_model": record.get("generator"),
        "final_verdict": {
            "verdict": record.get("final_verdict"),
            # where the label came from, and the reference beside it (§4.17 R5)
            "verdict_status": record.get("verdict_status"),
            "confidence": record.get("confidence"),
            "posterior": record.get("posterior") if record.get("evidence") else None,
            "reference": record.get("reference"),
            "policy_reasons": record.get("policy_reasons"),
        },
        "conversations": render_conversation(record, record.get("_trace"),
                                             observation_text),
        "evidence_chain": record.get("evidence") or [],
        "metadata": {
            "schema_version": SCHEMA_VERSION,
            "task_type": "fully_generated_image_detection",
            "evidence_scope": "global",
            "region_semantics": "diagnostic_evidence_region",
            "trajectory_id": record["trajectory_id"],
            "policy": record.get("policy"),
            "model_candidate": record.get("model_candidate"),
            "tools_served": record.get("tools_served"),
            "treatment": record.get("treatment"),
            "split": record.get("split"),
            "admission_category": category,
            "counters": record.get("counters"),
            "scores": record.get("scores"),
            "source_resolution": (split_entry or {}).get("resolution"),
        },
        "type": category,
        "source": "g4_admitted",
        # Fidelity, recorded rather than assumed (§4.17 R6): whether the
        # reasoning section carries the model's own argument verbatim, and
        # what the whole-dialogue audit found in the supervised turns.
        "fidelity": {
            "model_reasoning_verbatim": bool(model_reasoning),
            "reasoning_source": "model+bookkeeping" if model_reasoning else "bookkeeping_only",
            "dialogue_audit": {
                "supervised_turns": dialogue["turns"],
                "defect_count": len(dialogue["defects"]),
                "candidate_count": len(dialogue["candidates"]),
                "defect_kinds": sorted({finding["kind"] for finding in dialogue["defects"]}),
                # turns that re-requested a measurement already in hand: kept
                # in the record, never presented as a model action to copy
                "repeat_request_turns": dialogue["repeat_request_turns"],
            },
        },
        "audit": {"review_status": "pending_human", "review_method": "automated_validator"},
    }
    return sample


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def parse_sections(answer: str) -> Dict[str, str]:
    found = {}
    for name, body in _SECTION_PATTERN.findall(answer or ""):
        found[name] = body
    return found


def _container_reason_sentences(text: str) -> List[str]:
    """Sentences that use the container format as a reason for the label."""
    offenders = []
    for sentence in _SENTENCE_SPLIT.split(text or ""):
        lowered = sentence.lower()
        if not any(w in lowered for w in CONTAINER_WORDS):
            continue
        if not any(w in lowered for w in DIRECTION_WORDS):
            continue
        if not any(c in lowered for c in CAUSAL_CONNECTIVES):
            continue
        if any(d in lowered for d in CONTAINER_DENIALS):
            continue
        offenders.append(sentence.strip()[:120])
    return offenders


def validate_sample(sample: dict, split: Optional[dict] = None,
                    applicability: Optional[Dict[str, Dict[str, float]]] = None,
                    threshold: float = 0.65) -> List[str]:
    """
    Every check, as a list of problems (empty means the sample passes).

    `split` and `applicability` are injected so this can run without touching
    the real artifacts; callers that have them pass them.
    """
    problems: List[str] = []
    metadata = sample.get("metadata") or {}
    record = {
        **sample,
        "evidence": sample.get("evidence_chain") or [],
        "policy_reasons": (sample.get("final_verdict") or {}).get("policy_reasons"),
        # which policy produced this sample decides which verdict rules apply
        "policy_version": metadata.get("policy_version") or sample.get("policy_version"),
        "verdict_status": (sample.get("final_verdict") or {}).get("verdict_status"),
        "reference": (sample.get("final_verdict") or {}).get("reference"),
        "posterior": (sample.get("final_verdict") or {}).get("posterior"),
        "confidence": (sample.get("final_verdict") or {}).get("confidence"),
        "final_verdict": (sample.get("final_verdict") or {}).get("verdict"),
        "resolution": (metadata.get("source_resolution")
                       or [(sample.get("evidence_chain") or [{}])[0].get("image_height")]),
    }

    # 1. source/partition leakage
    if split is not None:
        source_id = (metadata.get("trajectory_id") or "").split("__")[-1]
        entry = next((e for key, e in split["sources"].items()
                      if key.replace("/", "_") in source_id), None)
        if entry is None:
            problems.append("leakage: sample source not found in the split")
        elif entry["split"] != "train":
            problems.append(f"leakage: source belongs to partition {entry['split']}")

    # 1b. no two conclusions in a row — the correction that separated them
    #     would have been dropped, and the model would be trained to reverse
    #     itself with nothing in between saying why.
    conversation = sample.get("conversations") or []
    for index, turn in enumerate(conversation[:-1]):
        if (turn.get("from") == "gpt" and "<verdict>" in (turn.get("value") or "")
                and conversation[index + 1].get("from") == "gpt"):
            problems.append(
                f"structure: assistant turn {index} states a conclusion and is "
                f"followed by another assistant turn — the correction that the "
                f"model was answering is missing")

    # 2. structure
    # The training target is the *last* assistant turn: earlier ones are the
    # tool calls that delivered the evidence.
    answer = next((t["value"] for t in reversed(sample.get("conversations", []))
                   if t.get("from") == "gpt"), "")
    sections = parse_sections(answer)
    for name in SECTIONS:
        if name not in sections:
            problems.append(f"structure: missing <{name}>")
    if len(SECTIONS) == len(sections):
        order = [m[0] for m in _SECTION_PATTERN.findall(answer)]
        if order != list(SECTIONS):
            problems.append(f"structure: sections out of order: {order}")
    verdict_body = sections.get("verdict", "{}")
    try:
        verdict = json.loads(verdict_body)
    except ValueError:
        verdict = None
        problems.append("structure: <verdict> is not valid JSON")
    if verdict is not None:
        if verdict.get("verdict") not in ("Real", "Fake", "Uncertain"):
            problems.append(f"structure: illegal label {verdict.get('verdict')!r}")
        confidence = verdict.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            problems.append(f"structure: confidence out of range: {confidence}")

    # 3. referenced evidence ids exist
    referenced = re.findall(r"evidence_id:\s*(\S+)", sections.get("forensic_evidence", ""))
    known = [t.get("evidence_id") for t in record["evidence"]]
    for identifier in referenced:
        if identifier not in known:
            problems.append(f"evidence: unknown evidence_id {identifier}")
    if len(set(referenced)) != len(referenced):
        problems.append("evidence: the same evidence_id is referenced twice")

    # 4. measurement scope and region bounds
    # A sample without a usable resolution skips the bounds check rather than
    # taking the whole validation run down with it.
    height = width = None
    resolution = record.get("resolution")
    if isinstance(resolution, (list, tuple)) and len(resolution) >= 2:
        height, width = resolution[0], resolution[1]
    for token in record["evidence"]:
        if token.get("measurement_scope") != "global":
            problems.append(f"scope: {token.get('evidence_id')} is not a global measurement")
        region = token.get("region_pixels")
        if height and width and region:
            ymin, xmin, ymax, xmax = region
            if not (0 <= ymin < ymax <= height and 0 <= xmin < xmax <= width):
                problems.append(f"region: {token.get('evidence_id')} bbox outside the image")

    # 5. duplicate invocations
    sources = [t.get("source") for t in record["evidence"]]
    duplicates = {s for s in sources if sources.count(s) > 1}
    if duplicates:
        problems.append(f"duplicates: repeated expert calls {sorted(duplicates)}")

    # 6. direction consistency (calibration is the authority)
    from state_machine.evidence_rectifier import EvidenceRectifier
    from utils.evidence_consistency import EvidenceConsistencyChecker

    for token in record["evidence"]:
        direction = EvidenceRectifier.direction_from_calibration(token)
        if direction is not None and token.get("support") != direction:
            problems.append(
                f"direction: {token.get('evidence_id')} says {token.get('support')} "
                f"but the calibration gives {direction}")
        check = EvidenceConsistencyChecker.check(token)
        if not check["ok"]:
            problems.append(f"direction: {token.get('evidence_id')} fails the gate: {check['failures']}")

    # 7. applicability of every expert used
    if applicability is not None:
        from scripts.generate_trajectories_g4 import tool_applicable

        from scripts.admit_trajectories_g4 import CORROBORATION_ONLY

        key = {"frequency_expert_v2": "freq", "noise_expert": "noise",
               "jpeg_expert": "jpeg", "probe_dino_expert": "probe"}
        carrying_passed = False
        deferred: List[str] = []
        for token in record["evidence"]:
            expert = token.get("source")
            if expert in RETIRED_EXPERTS:
                problems.append(f"applicability: retired expert {expert}")
                continue
            tool = key.get(expert)
            if tool is None:
                problems.append(f"applicability: unknown expert {expert}")
                continue
            ok, why = tool_applicable(tool, metadata.get("treatment"), applicability, threshold)
            if ok and tool not in CORROBORATION_ONLY:
                carrying_passed = True
            elif not ok and tool not in CORROBORATION_ONLY:
                problems.append(f"applicability: {why}")
            elif not ok:
                # Mirrors admission: a corroboration-only expert below the bar
                # is acceptable only because something else carries the call.
                deferred.append(why)
        # Weak evidence that ends in an honest abstention is exactly the
        # lesson we want to keep; the bar applies to labels, not to silence.
        if deferred and not carrying_passed and record["final_verdict"] in ("Real", "Fake"):
            problems += [f"applicability: {why}" for why in deferred]

    # Tool usage is declared by the policy, not inferred from whether evidence
    # came back: a tool session that produced none is a defect to report, not a
    # tool-free sample to wave through.
    uses_tools = bool(metadata.get("tools_served"))
    if not uses_tools:
        # A tool-free answer is the model's own; the posterior is the prior and
        # the evidence requirement does not apply.  What must hold is that the
        # sample reports the conclusion the model actually reached.
        candidate = metadata.get("model_candidate")
        if (record["final_verdict"] in ("Real", "Fake")
                and candidate and record["final_verdict"] != candidate):
            problems.append(
                f"verdict: no-tool sample claims {record['final_verdict']} "
                f"but the model said {candidate}")

    # 8. verdict agrees with its own posterior (evidence-based conclusions)
    #
    # Under v3 the posterior is a *reference*: the label is the model's own
    # conclusion and its confidence is the model's own statement, so a
    # disagreement with the reference is recorded, not an error.  These rules
    # bound the v2 records, where the posterior really did decide the label —
    # applying them to v3 samples would call every model-led decision wrong.
    posterior = record.get("posterior") if uses_tools else None
    model_led = (record.get("policy_version") == "v3"
                 or record.get("verdict_status") in ("model_concluded",
                                                     "no_parseable_conclusion"))
    if posterior is not None and not model_led:
        expected = "Fake" if posterior >= 0.5 else "Real"
        if record["final_verdict"] in ("Real", "Fake") and record["final_verdict"] != expected:
            problems.append(
                f"verdict: {record['final_verdict']} contradicts posterior {posterior}")
        if record["final_verdict"] == "Uncertain" and abs(posterior - 0.5) >= 0.34:
            problems.append(
                f"verdict: abstained although the posterior is {posterior:.2f}")
        # The reported confidence is P(the label we report), not the posterior:
        # a Real verdict carrying P(Fake) is what the review found in 12 samples.
        reported = record.get("confidence")
        if record["final_verdict"] in ("Real", "Fake") and reported is not None:
            expected_confidence = (posterior if record["final_verdict"] == "Fake"
                                   else 1.0 - posterior)
            if abs(float(reported) - expected_confidence) > 1e-3:
                problems.append(
                    f"confidence: {record['final_verdict']} reported at {reported} "
                    f"but P({record['final_verdict']}) is "
                    f"{round(expected_confidence, 4)}")
    if model_led:
        # What still has to hold: the confidence is a number in range, and the
        # disagreement (if any) is on the record rather than hidden.
        reported = record.get("confidence")
        if reported is not None and not 0.0 <= float(reported) <= 1.0:
            problems.append(f"confidence: {reported} 不在 [0,1]")
        if record.get("verdict_status") == "no_parseable_conclusion":
            problems.append("verdict: 结案不可解析，不能作为训练样本")
        reference = record.get("reference") or {}
        if (posterior is not None and record["final_verdict"] in ("Real", "Fake")
                and ("Fake" if posterior >= 0.5 else "Real") != record["final_verdict"]
                and "reference_disagrees" not in (reference.get("advisory") or [])):
            problems.append("verdict: 与参考后验不一致但未在 reference.advisory 记录")

    # 9. confident labels need admissible evidence
    confidence = record.get("confidence") or 0.0
    admissible = [t for t in record["evidence"]
                  if t.get("support") in ("Real", "AI-generated")
                  and (t.get("applicability") or "").startswith(("strong", "weak", "inverted"))]
    if (uses_tools and record["final_verdict"] in ("Real", "Fake")
            and confidence >= CONFIDENT_THRESHOLD and not admissible):
        problems.append("confidence: label is confident but no evidence supports it")

    # 10. the conversation delivers the evidence its answer cites
    turns = sample.get("conversations") or []
    evidence_turns = [t for t in turns
                      if t.get("from") == "user" and str(t.get("value", "")).lstrip().startswith("{")]
    if turns and turns[0].get("from") != "user":
        problems.append("conversation: must open with a user turn")
    if turns and turns[-1].get("from") != "gpt":
        problems.append("conversation: must close with the assistant answer")
    if uses_tools:
        if len(evidence_turns) != len(record["evidence"]):
            problems.append(
                f"conversation: {len(evidence_turns)} evidence turns for "
                f"{len(record['evidence'])} tokens — the answer cites evidence "
                f"the model was never given")
        for turn in evidence_turns:
            try:
                payload = json.loads(turn["value"])
            except ValueError:
                problems.append("conversation: an evidence turn is not valid JSON")
                continue
            if payload.get("evidence_id") not in known:
                problems.append(
                    f"conversation: turn delivers unknown evidence_id "
                    f"{payload.get('evidence_id')}")
    elif evidence_turns:
        problems.append("conversation: a tool-free sample carries evidence turns")

    # 11. the container is not a reason
    for offender in _container_reason_sentences(answer):
        problems.append(f"container_reason: {offender}")

    # 12. the assistant may not cite an expert it never ran (grounding)
    called = {t.get("source") for t in record["evidence"]}
    for index, turn in enumerate(turns):
        if turn.get("from") != "gpt":
            continue
        for tool, is_call in _tool_mentions(turn.get("value", "")):
            if is_call:
                continue        # an action the controller decides to execute
            if tool not in called:
                problems.append(
                    f"grounding: assistant turn {index} refers to {tool} results "
                    f"but that expert was never called")

    # 13. a global measurement may not be described as a local one
    for token in record["evidence"]:
        if token.get("measurement_scope") != "global":
            continue
        for field in ("phenomenon", "reasoning", "interpretation_text"):
            if _LOCAL_MEASUREMENT.search(str(token.get(field) or "")):
                problems.append(
                    f"scope: {token.get('evidence_id')} is a global measurement but "
                    f"its {field} describes a local one")

    # 14. the observation has to say something about the image
    observation = sections.get("observation", "")
    content_lines = [line for line in observation.splitlines()
                     if line.strip() and not _METADATA_LINE.match(line.strip())]
    if not content_lines:
        problems.append(
            "observation: only container and resolution — nothing about the image "
            "that a reviewer could check")

    return problems


def validate_dataset(samples: Iterable[dict], **kwargs) -> Dict[str, List[str]]:
    """Problems per sample id, for every sample that has any."""
    findings: Dict[str, List[str]] = {}
    for sample in samples:
        problems = validate_sample(sample, **kwargs)
        if problems:
            findings[sample.get("id", "?")] = problems
    return findings


def load_split(path: Optional[str] = None) -> Optional[dict]:
    path = path or os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None
