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
_SENTENCE_SPLIT = re.compile(r"[。.;；\n]")


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


def render_observation(record: dict, observation_text: str = "") -> str:
    """What can be checked in the image itself, plus how it was delivered."""
    facts = [
        f"container: {record.get('container')}, quality: {record.get('quality', record.get('treatment'))}",
        f"resolution: {record.get('resolution')}",
    ]
    body = "\n".join(f"- {fact}" for fact in facts)
    if observation_text:
        body = observation_text.strip() + "\n" + body
    return body


def render_reasoning(record: dict) -> str:
    """Support, counter-evidence, why an expert may fail, what stays uncertain."""
    tokens = record.get("evidence") or []
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
    lines.append(
        f"剩余不确定性：后验 P(Fake)={record.get('posterior')}，"
        f"冲突度={record.get('conflict_score')}，停止原因={'/'.join(record.get('policy_reasons') or [])}"
    )
    return "\n".join(lines)


def render_answer(record: dict, observation_text: str = "") -> str:
    """The structured assistant turn final_v2 trains on."""
    tokens = record.get("evidence") or []
    evidence_blocks = "\n".join(render_evidence_block(t) for t in tokens)
    verdict = {
        "verdict": record.get("final_verdict"),
        "confidence": record.get("confidence"),
        # A tool-free conclusion rests on the model's perception, not on an
        # evidence posterior — which is 0.5 by construction and says nothing.
        "basis": "model_perception" if not tokens else "evidence_posterior",
    }
    if tokens:
        verdict["posterior"] = record.get("posterior")
    return (
        f"<observation>\n{render_observation(record, observation_text)}\n</observation>\n\n"
        f"<forensic_evidence>\n{evidence_blocks or '- 本轨迹未调用工具'}\n</forensic_evidence>\n\n"
        f"<reasoning>\n{render_reasoning(record)}\n</reasoning>\n\n"
        f"<verdict>\n{json.dumps(verdict, ensure_ascii=False)}\n</verdict>"
    )


def render_sample(record: dict, category: str, observation_text: str = "",
                  split_entry: Optional[dict] = None) -> dict:
    """Assemble a final_v2 sample from one admitted trajectory."""
    image_path = record.get("image_path") or record.get("variant_path", "")
    sample = {
        "id": f"f2_{record['trajectory_id']}",
        "image_path": image_path,
        "ground_truth": record["ground_truth"],
        "source_model": record.get("generator"),
        "final_verdict": {
            "verdict": record.get("final_verdict"),
            "confidence": record.get("confidence"),
            "posterior": record.get("posterior") if record.get("evidence") else None,
            "policy_reasons": record.get("policy_reasons"),
        },
        "conversations": [
            {"from": "user", "value": "<image>\n请分析这张图像的真实性。"},
            {"from": "gpt", "value": render_answer(record, observation_text)},
        ],
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

    # 2. structure
    answer = next((t["value"] for t in sample.get("conversations", [])
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
    height, width = (record["resolution"] or [None, None])[:2] if isinstance(
        record["resolution"], list) else (None, None)
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

        key = {"frequency_expert_v2": "freq", "noise_expert": "noise", "jpeg_expert": "jpeg"}
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
    posterior = record.get("posterior") if uses_tools else None
    if posterior is not None:
        expected = "Fake" if posterior >= 0.5 else "Real"
        if record["final_verdict"] in ("Real", "Fake") and record["final_verdict"] != expected:
            problems.append(
                f"verdict: {record['final_verdict']} contradicts posterior {posterior}")
        if record["final_verdict"] == "Uncertain" and abs(posterior - 0.5) >= 0.34:
            problems.append(
                f"verdict: abstained although the posterior is {posterior:.2f}")

    # 9. confident labels need admissible evidence
    confidence = record.get("confidence") or 0.0
    admissible = [t for t in record["evidence"]
                  if t.get("support") in ("Real", "AI-generated")
                  and (t.get("applicability") or "").startswith(("strong", "weak", "inverted"))]
    if (uses_tools and record["final_verdict"] in ("Real", "Fake")
            and confidence >= CONFIDENT_THRESHOLD and not admissible):
        problems.append("confidence: label is confident but no evidence supports it")

    # 10. the container is not a reason
    for offender in _container_reason_sentences(answer):
        problems.append(f"container_reason: {offender}")

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
