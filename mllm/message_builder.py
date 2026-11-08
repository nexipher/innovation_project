"""
Qwen2.5-VL chat-message construction (plan.md §4.8 G1).

Kept import-light (PIL + stdlib only) so the conversation protocol can be
unit-tested on CPU without loading transformers/torch.

Multi-turn image protocol:
  - the original image stays attached to the first user turn;
  - every evidence turn may carry `image_paths` (diagnostic region crops),
    which are attached as image blocks before the evidence text;
  - later turns never drop earlier images — the full history is rebuilt on
    every generate() call.
"""

import os
from typing import Dict, List, Optional

from PIL import Image

from config import PROJECT_ROOT

"""Qwen2.5-VL chat-message construction (plan.md §4.8 G1).

Kept import-light (PIL + stdlib only) so the conversation protocol can be
unit-tested on CPU without loading transformers/torch.

Multi-turn image protocol:
  - the original image stays attached to the first user turn;
  - every evidence turn may carry `image_paths` (diagnostic region crops),
    which are attached as image blocks before the evidence text;
  - later turns never drop earlier images — the full history is rebuilt on
    every generate() call.

The forensic prompt is assembled per tool policy (G4-b): a trajectory that may
call only the noise expert must not be told that three experts exist, or the
model spends turns requesting tools the session will not serve.  The default
policy reproduces the shipped all-tools prompt byte for byte.
"""

import os
from typing import Dict, Iterable, List, Optional

from PIL import Image

from config import PROJECT_ROOT

# Per-tool prompt fragments: the action the model may emit, and what that tool
# was measured to do on the format-balanced calibration set (G2-b).
TOOL_ACTIONS: Dict[str, str] = {
    "probe": "- <call_probe>[ymin, xmin, ymax, xmax]</call_probe> → call AI-generation detector (whole-image score)",
    "freq": "- <call_freq>[ymin, xmin, ymax, xmax]</call_freq>  → call frequency-domain expert",
    "noise": "- <call_noise>[ymin, xmin, ymax, xmax]</call_noise> → call noise residual expert",
    "jpeg": "- <call_jpeg>[ymin, xmin, ymax, xmax]</call_jpeg>   → call JPEG compression expert",
    "trufor": "- <call_trufor>[ymin, xmin, ymax, xmax]</call_trufor> → call local manipulation/localization expert",
}

TOOL_MEASURES: Dict[str, str] = {
    "probe": "- <call_probe>  AI-generation discriminator over the whole image. Its score is\n  the model's predicted probability for the AI-generated/fake class (official\n  classification threshold 0.5; HIGH points to Fake, LOW points to Real), but it is\n  NOT calibrated on this project's target data. It measures the image as a whole and\n  locates NOTHING — never restate it as a physical artifact or as an area of the\n  image; use localization evidence for regions and coverage.",
    "freq": "- <call_freq>  multi-scale frequency analysis. WEAK evidence (separation 0.56-0.63,\n  barely above chance). Use it to corroborate, never as the deciding signal.",
    "noise": "- <call_noise> residual micro-noise LEVEL. COUNTER-INTUITIVE: a HIGH level points to\n  Real (camera sensor micro-noise) and a LOW level points to Fake (smooth generator\n  output). This is stable across image formats. Do NOT read a high value as forgery.",
    "jpeg": "- <call_jpeg>  JPEG compression history (blockiness / DCT structure). A HIGH level\n  points to Real (a camera JPEG that was saved or re-saved), NOT to forgery. After\n  heavy re-compression (quality <= 70) this measurement is near chance and must be\n  ignored.",
    "trufor": "- <call_trufor>  local manipulation/localization. It returns a detector score plus\n  an anomaly map and a confidence map aligned to the ORIGINAL image, and its bbox\n  is a diagnostic region in [0,1000]. A high score/high anomaly map is evidence of\n  LOCAL manipulation, NOT a whole-image AI-generation score; the confidence map\n  says where the map is reliable. Do not use it to answer the whole-image question,\n  and do not treat a bright region as proof without checking its confidence and\n  the original image.",
}

# 登记表支持的工具有序表；freq/jpeg/noise 是待退役的旧工具，保留其手写文本供
# 历史复算与 v2 对照装配提示词（阶段 4.4：提示词由登记表生成，不再手写新工具）。
TOOL_ORDER = ("probe", "npr", "univfd", "noise", "freq", "jpeg")
EXTRA_TOOLS = ("trufor",)
ALL_TOOL_TAGS = TOOL_ORDER + EXTRA_TOOLS
REGISTRY_TAGS = ("probe", "npr", "univfd", "noise")


def _registry_entries() -> Dict[str, dict]:
    """variant（probe/npr/univfd/noise）→ 登记表条目。"""
    try:
        from utils.score_semantics import load
    except Exception:                     # 登记表不可用时不阻断历史装配
        return {}
    out = {}
    for _name, record in load()["experts"].items():
        variant = record.get("variant")
        if variant:
            out[variant] = record
    return out


def _registry_tool_text(tag: str, record: dict) -> str:
    """
    一个工具的动作行 + 它到底测什么 + 该怎么读——全部来自登记表。

    手写表就是漂移的来源：工具换了、条件变了，提示词还在说旧话。这里每个字段
    只有一个出处（`experts/score_semantics.json`），改一处就够。
    """
    semantics = record["raw_score"]
    threshold = record["threshold"]
    scope = record["preprocess"]["measurement_scope"]
    direction = {
        "higher_is_fake": "HIGH points to Fake, LOW points to Real",
        "higher_is_real": "COUNTER-INTUITIVE: HIGH points to Real, LOW points to Fake",
        "none": "no direction",
    }[semantics["direction"]]
    action = (f"- <call_{tag}>[ymin, xmin, ymax, xmax]</call_{tag}> → {record['display_name']}")
    lines = [
        action,
        f"  Range {semantics['range'][0]}–{semantics['range'][1]}, threshold "
        f"{threshold['value']} (chosen on: {threshold['chosen_on']}). {direction}.",
        f"  It is NOT a calibrated probability; do not average it with another expert's score.",
        f"  Scope: {scope} — " + (
            "it measures the WHOLE image and locates nothing."
            if scope == "global" else
            "it does NOT see the image edges; do not read it as a whole-image measurement."),
    ]
    # 只投喂给模型看的那一句；内部说明（含统计数值与写法备注）不进提示词。
    if record.get("prompt_note"):
        lines.append(f"  Condition: {record['prompt_note']}")
    return "\n".join(lines)

PROMPT_HEAD = "You are an AI forensic image analyst. You MUST follow this EXACT format in EVERY response. Do NOT write free-form analysis.\n\n"
PROMPT_MID = (" Do NOT use <call_call_freq>, <call_call_noise>, <call_call_probe>, <call_frequency>, or any other variation.\n\n"
    "ONE ACTION PER RESPONSE (mandatory): exactly ONE of the two structures below.\n"
    "Never a call together with a verdict, never two calls in one response. A response\n"
    "carrying more than one action is refused unexecuted, and the turn is wasted.\n\n"
    "COORDINATES: bbox values are integers in range [0, 1000], format [ymin, xmin, ymax, xmax].\n"
    "These are NORMALIZED image coordinates, not pixels: pixel width/height, canvas\n"
    "size and content-box values are never bbox values. A bbox equal to the image's\n"
    "pixel dimensions is invalid; use [0,0,1000,1000] only for the whole frame.\n"
    "The bbox is OPTIONAL. Every expert measures the WHOLE image, so a region only marks\n"
    "what you find visually interesting; it does not restrict the measurement. If nothing\n"
    "stands out, call the expert once without a bbox — do NOT invent a suspicious region.\n\n"
    "RESPONSE FORMAT (MANDATORY — every response must contain one of these two structures):\n\n"
    "Structure A — When you need forensic evidence:\n"
    "<planning>\n"
    "Suspected Region: [ymin, xmin, ymax, xmax] or none\n"
    "Visual Observation: [describe the scene and the main subject objectively — what is in the\n"
    "  picture, its texture, edges, lighting and colour; this is description, not a verdict]\n"
    "Suspected Anomalies: [what, if anything, looks off; if nothing stands out, say so plainly.\n"
    "  Never invent noise, compression blocks or sensor artifacts you cannot see — the experts\n"
    "  measure those; the generic sentence alone is not a description]\n"
    "Expert Target & Hypothesis: [the expert's NAME only, e.g. probe — and why;\n"
    "  a <call_x> tag here is read as a second request and the whole turn is refused]\n"
    "</planning>\n"
    "<call_EXPERT_NAME>[ymin, xmin, ymax, xmax]</call_EXPERT_NAME>\n"
    "(the bbox may be left out entirely: <call_EXPERT_NAME></call_EXPERT_NAME>)\n"
    "The call tag appears EXACTLY ONCE, AFTER </planning>. Never write a <call_x> tag\n"
    "inside the planning block, and never two tags in one response.\n\n"
    "Structure B — When you have enough evidence to conclude:\n"
    "<reasoning>\n"
    "[Cross-reference the expert findings with your visual observations. When the evidence\n"
    " points in different directions, name the conflicting experts, their readings and the\n"
    " limits those readings carry, then decide from the evidence whether to call another\n"
    " expert, conclude or abstain. Never invoke a \"presumption of innocence\" instead of\n"
    " analysing the evidence. Describe visible artefacts as what you see (blockiness, blur);\n"
    " do not assert their cause, and take image dimensions from the input metadata rather\n"
    " than from appearance. State how many experts returned a valid measurement and whether\n"
    " you compared them across experts.]\n"
    "</reasoning>\n"
    "<verdict>\n"
    "{\"verdict\": \"Real\"|\"Fake\"|\"Uncertain\", \"confidence\": 0.0-1.0, \"primary_evidence\": [\"evidence_name1\"], \"report\": \"concise forensic report in Chinese\"}\n"
    "(confidence = your own uncalibrated certainty in the FINAL verdict, based on evidence\n"
    " quality, expert applicability and unresolved questions; not P(Real/Fake): RULES 9)\n"
    "</verdict>\n\n")

# Legacy example line kept separate so the joint mode can replace exactly this
# schema without rewriting the rest of the protocol.
LEGACY_VERDICT_JSON = (
    '{\"verdict\": \"Real\"|\"Fake\"|\"Uncertain\", \"confidence\": 0.0-1.0, '
    '\"primary_evidence\": [\"evidence_name1\"], \"report\": \"concise forensic '
    'report in Chinese\"}')
JOINT_VERDICT_JSON = (
    '{"verdict": "Real"|"Fake"|"Uncertain", '
    '"whole_image_generation": "Real"|"AI-generated"|"Uncertain", '
    '"local_manipulation": "Tampered"|"Not tampered"|"Uncertain"|"Not applicable", '
    '"suspicious_regions": [{"bbox": [ymin,xmin,ymax,xmax], "reason": "what local '
    'evidence supports this region", "evidence": ["evidence_id1"]}], '
    '"confidence": 0.0-1.0, "primary_evidence": ["evidence_name1"], '
    '"report": "concise forensic report in Chinese"}')
JOINT_GUIDANCE = (
    "\nJOINT TASK (this session serves the localisation expert): keep TWO answers "
    "separate. `whole_image_generation` answers whether the image as a whole is "
    "Real / AI-generated / Uncertain; PROBE is the whole-image witness. "
    "`local_manipulation` answers whether there is local editing (Tampered / Not "
    "tampered / Uncertain / Not applicable); TruFor's score and maps are the local "
    "witness. The overall `verdict` is YOUR synthesis of both answers and both "
    "evidence bases; do not derive it by a fixed rule from the two dimensions. "
    "`Not tampered` means this examination found no evidence sufficient to support "
    "local tampering; it is not proof that nothing was altered. `Not applicable` is "
    "only for a whole image judged AI-generated, where the local question does not "
    "apply. `Uncertain` is not a generic abstention: name which question is "
    "uncertain, what limits the evidence (failed/missing measurement, applicability, "
    "low reliability, image-text mismatch), and what information is missing. Do not "
    "use 'cannot prove it was not tampered' as the whole reason, and do not turn a "
    "missing measurement into Not tampered. Put local suspicions in "
    "`suspicious_regions` with original-image [0,1000] bbox, a concrete reason, and "
    "the evidence_id(s) that support them; use an empty list when there is no local "
    "suspicion. A full-image bbox means the global range, not a localized finding. "
    "A bright map region is a candidate, not proof; a low whole-image PROBE score "
    "does not clear a local map, and a high local map does not prove whole-image AI "
    "generation. When both detectors return high values, explain what each "
    "measurement contributes; do NOT average them and do NOT treat the coincidence "
    "as mutual verification. Call order is yours: if your initial observation gives "
    "a concrete local suspicion, you may call TruFor first, then decide whether "
    "PROBE is needed for the whole-image question. Do not follow a fixed "
    "PROBE -> TruFor -> conclude sequence. Before a verdict, check both questions "
    "explicitly: which has a valid measurement, which is unresolved, and whether an "
    "untried served expert could change the answer. If it could, and the budget "
    "allows, call it. If you conclude with one expert, explain why the other "
    "measurement would not materially change the final conclusion. Do not cite "
    "'X was not called' as the reason for Uncertain while X is still callable; if X "
    "failed, treat that gap separately from the other question and check whether the "
    "remaining expert can still answer part of it. TruFor artifacts: the original "
    "crop, anomaly map and confidence map show the SAME content region at the same "
    "content size; titles and colour legends are outside the content. In the anomaly "
    "map blue is low and red is high response; in the confidence map blue is low and "
    "red is high reliability. For that same region, state separately: (a) what is "
    "visible in the original, (b) what the anomaly response is there, (c) what the "
    "reliability is there, and (d) how that combination supports or limits the local "
    "conclusion. Do not transfer a red background in the reliability map onto the "
    "object; check the object's own location. Do not assert a specific editing "
    "process (for example 'text was added later') from an anomaly map. If you "
    "call TruFor without a bbox, the system shows an anomaly-peak inspection crop "
    "from the existing map; it is not a new measurement and not a predicted bbox. "
    "If the panels show the full frame, do NOT use [0,0,1000,1000] as a local "
    "suspicious region: leave suspicious_regions empty and say that no localized "
    "region was established. Bboxes are normalized [0,1000]; pixel dimensions and "
    "layout values are never bboxes. "
    "`raw_band` is only a five-bin reading aid for the raw score: the middle band "
    "needs more visual/other-measurement analysis; it is not a label or a "
    "confidence, and the same band in two experts does not mean equal evidence.\n")
JOINT_RULE_10 = (
    "\n10. When both PROBE and TruFor are served, do not merge their questions: "
    "PROBE speaks to whole-image generation; TruFor speaks to local manipulation "
    "and returns same-region map/conf panels in original-image coordinates. A map "
    "bright spot is a local candidate, not proof; for the same content location, "
    "state the original content, the anomaly response, and the reliability "
    "separately, give the region in normalized [0,1000], and cite the evidence_id. "
    "Do not transfer a red background in the reliability map onto the object. Do "
    "not assert a specific editing process from the map. Do not average the two "
    "scores or call simultaneous high scores mutual verification.\n")

PROMPT_TAIL_BASE = (
    "\nEach evidence token you receive has EXACTLY these fields:\n"
    "  `evidence_id`, `expert`, `measurement`, `status`, `scope`,\n"
    "  `score` = {`name`, `value`, `range`, `direction`, `direction_for_this_value`,\n"
    "             `threshold`, `threshold_chosen_on`, `raw_band`,\n"
    "             `is_calibrated_probability`: always false},\n"
    "  `reading` = a short account of what was measured and the conditions under which\n"
    "              the number may be read,\n"
    "  `artifact_layout` (when maps are attached) = content-area label and\n"
    "             colour-legend names (no layout value is a bbox),\n"
    "  and, when the expert rendered one, `visual_artifacts` + `artifact_meaning`.\n"
    "\n1. There is ONE score per token. Its `direction` says which side means what —\n"
    "   follow it, never your own expectation about the expert. Scores from different\n"
    "   experts are NOT comparable and must not be averaged or ranked against each other.\n"
    "2. Read the `reading` before using the number: it states the conditions. `direction`\n"
    "   is the expert's scale; `direction_for_this_value` is what THIS number means under\n"
    "   those conditions. When it is `Uncertain`, the number carries no direction here:\n"
    "   say so instead of forcing one.\n"
    "3. PROBE's score is the model's predicted probability for the AI-generated/fake\n"
    "   class (official classification threshold 0.5; higher = more generated), but it\n"
    "   is NOT calibrated on this project's target data. Other experts' scores may not\n"
    "   be probabilities. Never present any raw score as a calibrated probability.\n"
    "4. `status` tells you whether a measurement happened: a `failed` or missing\n"
    "   measurement is NOT a neutral result, and must never be reported as one. State\n"
    "   which measurement is missing and how that gap limits the current conclusion.\n"
    "   If another served tool can still measure that question and the budget allows,\n"
    "   call it before treating the gap as final.\n"
    "5. `scope` says what the expert could see: `global` = the whole image;\n"
    "   `center_crop` = it never saw the edges. Do not restate a crop-based measurement\n"
    "   as a whole-image finding, and do not restate a whole-image score as describing\n"
    "   an area.\n"
    "6. `visual_artifacts` are expert-produced images delivered with this token.\n"
    "   TruFor delivers an original crop, an anomaly-map crop and a confidence-map\n"
    "   crop of the SAME content region at the same content size; titles and colour\n"
    "   bars are outside the content. For that region, state separately: what is\n"
    "   visible in the original, what the anomaly response is there, what the\n"
    "   reliability is there, and how that combination supports or limits the\n"
    "   conclusion. Do not transfer a red background in the reliability map onto\n"
    "   the object; check the object's own location. In the anomaly map, blue = low\n"
    "   response and red = high response. In the confidence map, blue = low\n"
    "   reliability and red = high reliability. A red/bright region is a LOCAL\n"
    "   candidate, not proof; low confidence means do not lean on that area. Do not\n"
    "   assert a specific editing process (for example 'text was added later') from\n"
    "   an anomaly map. Artifacts do not replace the score and are not calibrated\n"
    "   probabilities. Predicted boxes are normalized original-image [0,1000];\n"
    "   never copy a pixel dimension, canvas size or layout value into a bbox.\n"
    "   If the panels are a full-frame context view, do not use\n"
    "   [0,0,1000,1000] as a local suspicious region; leave suspicious_regions\n"
    "   empty and say no localized region was established. If they are an\n"
    "   anomaly-peak inspection crop, it is a candidate from the existing map,\n"
    "   not a new measurement and not a predicted bbox.\n"
    "7. `raw_band` is a five-bin reading aid: [0,0.2) low, [0.2,0.4) lower,\n"
    "   [0.4,0.6) middle, [0.6,0.8) higher, [0.8,1] high. The middle band needs\n"
    "   more visual and other-measurement analysis. It does not map directly to a\n"
    "   label or confidence, and the same band across experts does not mean equal\n"
    "   evidence strength.\n"
)

# Reading rules that only make sense when the tool is served.  A session
# without PROBE must not be told how to read PROBE's token: the prompt's job is
# to describe this session, and the model is never shown a tool it cannot call.
#
# 这里**没有 probe**：登记表标签（`REGISTRY_TAGS`）的说明由 `_registry_tool_text`
# 生成，而本表在拼接时明确过滤掉这些标签——所以 probe 的条目曾经是一段永远不
# 生效的死代码（写在那儿，模型永远看不到）。它唯一有用的那句（PROBE 的分数不是
# 校准概率）已搬进登记表的 `prompt_note`，那里才是模型真正读到的地方。
TOOL_READ_NOTES: Dict[str, str] = {}

PROMPT_TAIL_RULES = "\nRULES:\n1. Before you conclude, check which questions already have a valid measurement,\n   which are unresolved, and which served tools have not been tried. If an\n   unresolved question is material to the final verdict, and a served tool is\n   applicable and the budget allows, call it. Use as many measurements as the\n   unresolved questions require, but never repeat a measurement. A single-expert\n   conclusion is allowed only when you explain why the second measurement would\n   not materially change the final conclusion.\n2. Call each expert AT MOST ONCE per session. Every expert measures the WHOLE image,\n   so a second call — whatever region you name — returns the same measurement: the\n   system answers it with the numbers you already have, and the turn is wasted.\n   This is a hard rule, not advice.\n3. EVERY response carries exactly one action tag: one <call_x> (bbox optional) or\n   one <verdict>. A response with no tag, or with more than one action, is refused.\n4. NEVER fabricate evidence — only reference evidence tokens you have received.\n5. You decide when the evidence is sufficient to conclude. `Uncertain` is not a default\n   when a tool is still available: if an unresolved question could be answered by a served,\n   applicable tool and the budget allows, investigate before concluding. If the evidence is\n   weak, conflicting, or mostly `Uncertain`/`disabled`, you may still say \"Uncertain\" — an\n   honest Uncertain is preferred over a confident mistake — but it must describe the\n   limitation that remains after the feasible investigation, or explain why the\n   investigation cannot continue (failed tool, no applicable tool, budget/format limit).\n   Do not write \"I did not call X\" as the reason for Uncertain when X is still callable.\n   An Uncertain conclusion must say which question is uncertain, what limits the evidence\n   (failed/missing measurement, applicability, low reliability, image-text mismatch), and\n   what is missing. `Not tampered` means this examination found no evidence sufficient to\n   support local tampering; it is not proof of absence, and a missing measurement must not\n   become Not tampered.\n6. Describe only what you can SEE before a call — appearance, texture, edges, lighting.\n   You cannot see SRM residuals, DCT histograms or sensor noise in the picture itself;\n   make those claims only after the expert that measures them has returned a token.\n7. Every conclusion states the scope of the evidence: how many distinct experts returned\n   a VALID measurement, and whether you compared them ACROSS EXPERTS. Call it a cross-expert\n   comparison, never \"independent cross-validation\": two detectors can share training\n   sources or fail together. One measurement is not a comparison, and calling a second\n   expert does not by itself make the verification independent. When PROBE and TruFor\n   are both high, explain their separate contributions; do not average them or call the\n   coincidence mutual verification. State what the missing evidence limits only when it\n   bears on the conclusion.\n8. Visible artefacts are observations (blockiness, blur, banding): describe them, do not\n   assert their cause. An anomaly map shows a response, not an edit history; never state\n   which edit was performed (for example, text added later). Never claim that compression, re-encoding or sensor noise has been\n   measured, ruled out or shown not to affect a result — only the expert that measures\n   them may say anything about them, and only within its recorded `scope`. Take image\n   dimensions from the input metadata; never infer \"high resolution\" from appearance.\n9. `confidence` is YOUR OWN uncalibrated certainty in the final overall\n   verdict. It is NOT a probability that the image is real or AI-generated, not calibrated,\n   and not an accuracy. Base it on evidence quality, expert applicability, and unresolved\n   questions; never build it by copying, inverting, averaging or multiplying expert scores, and do not output a\n   fixed high value by habit. When the verdict is \"Uncertain\", `confidence` is your\n   certainty about abstaining — never P(Fake). The report must make the basis of the\n   confidence clear and consistent with the stated conclusion.\n"

SUBSET_NOTE = ("(In this session ONLY the tools listed above exist. Requests for any other "
               "tool will not be served, and spending a turn on one wastes the budget.)")


def build_forensic_prompt(allowed_tools: Optional[Iterable[str]] = None) -> str:
    """
    Assemble the forensic system prompt for a tool policy.

    Args:
        allowed_tools: tool keys ("freq" | "noise" | "jpeg") the session will
            serve.  None means all of them, reproducing the shipped prompt.
    """
    if allowed_tools is None:
        tools = list(TOOL_ORDER)
    else:
        requested = set(allowed_tools)
        tools = [t for t in ALL_TOOL_TAGS if t in requested]
    if not tools:
        raise ValueError("a forensic prompt needs at least one tool; use the "
                         "baseline prompt for a no-tool session")

    registry = _registry_entries()
    actions_list = []
    for tag in tools:
        record = registry.get(tag)
        if record is not None:
            actions_list.append(_registry_tool_text(tag, record))
        else:
            actions_list.append(TOOL_ACTIONS[tag])
    actions = "\n".join(actions_list)
    # The legacy default (probe/npr/univfd/noise/freq/jpeg) is the full
    # historical prompt and must not carry a "subset" banner.  Any policy that
    # omits one of those tools is a subset and says so; a dual policy that adds
    # trufor but omits npr/univfd is also a subset.
    if allowed_tools is not None and not set(TOOL_ORDER).issubset(set(tools)):
        actions = actions + "\n" + SUBSET_NOTE
    # 登记表已把"测什么、怎么读、什么条件"写进了 action 段，故旧的两段仅用于
    # 没有登记条目的旧工具（freq/jpeg），且只在被允许时才出现。
    #
    # 段头只在段里有内容时才写：当前工具组（probe/npr/univfd）全部由登记表生成，
    # 这一段的正文是空的——留下段头，就是在提示词里向模型描述一批它这次根本看不到
    # 的工具，而段头里原本还带着我们的阶段编号（"G7 fitting group"）。校准出处
    # 该说的部分（格式配平、带标注来源）留下，内部编号不留给模型。
    measures_text = "\n".join(TOOL_MEASURES[t] for t in tools if t not in REGISTRY_TAGS)
    measures = ("WHAT EACH TOOL ACTUALLY MEASURES (empirical calibration on this "
                "project's own format-balanced, labelled sources):\n"
                + measures_text + "\n\n") if measures_text else ""
    read_notes = "".join(TOOL_READ_NOTES[t] for t in tools
                         if t in TOOL_READ_NOTES and t not in REGISTRY_TAGS)

    mid = PROMPT_MID
    rules = PROMPT_TAIL_RULES
    if "trufor" in tools:
        mid = mid.replace(LEGACY_VERDICT_JSON, JOINT_VERDICT_JSON) + JOINT_GUIDANCE
        rules = rules + JOINT_RULE_10
    return (PROMPT_HEAD
            + "AVAILABLE ACTIONS (use EXACTLY these tag names — do NOT invent variations):\n"
            + actions + "\n\nFORBIDDEN:" + mid
            + measures + "HOW TO READ AN EVIDENCE TOKEN:"
            + PROMPT_TAIL_BASE + read_notes + rules)


FORENSIC_SYSTEM_PROMPT = build_forensic_prompt()

# A TruFor turn carries three images: the diagnostic region crop, the anomaly
# map and the confidence map.  Capping at two silently dropped the confidence
# map — the model then saw a bright anomaly map with no reliability signal.
MAX_REGION_IMAGES_PER_TURN = 3

# No-tool variant (G2-d condition 1): the RGB baseline must answer from the
# image alone, so tool calls are forbidden instead of merely unavailable.
# The no-tool protocol, in two versions that differ for one reason.
#
# The historical one is what the G2-d/G3-e arms were measured under; it asks
# for reasoning and a verdict, and its text is frozen — those comparisons are
# part of the record and a prompt edit would quietly re-open them.
#
# The training version adds an explicit <observation> section.  Without it the
# model's visual description lands wherever it likes (a <planning> block it was
# not asked for, or the top of its reasoning), and the converter that builds
# training samples then has nowhere to read it from — which is how 30 of the
# first batch's 56 no-tool sessions became metadata-only observations and were
# rejected by the validator.
BASELINE_SYSTEM_PROMPT = """You are an AI forensic image analyst. Decide whether the image is real (camera-captured) or AI-generated/tampered using ONLY your own visual inspection. You have NO access to external forensic tools in this session.

CRITICAL OUTPUT FORMAT RULES:
1. Do NOT output <planning> or <call_*> tags — tool calls are forbidden.
2. Respond with a short <reasoning> block and then the verdict JSON:
<reasoning>
[your visual observations supporting the judgement]
</reasoning>
<verdict>
{"verdict": "Real"|"Fake"|"Uncertain", "confidence": 0.0-1.0, "primary_evidence": [], "report": "concise report in Chinese"}
</verdict>
"""

TRAINING_BASELINE_PROMPT = """You are an AI forensic image analyst. Decide whether the image is real (camera-captured) or AI-generated/tampered using ONLY your own visual inspection. You have NO access to external forensic tools in this session.

CRITICAL OUTPUT FORMAT RULES:
1. Do NOT output <planning> or <call_*> tags — tool calls are forbidden.
2. Respond with THREE sections in this order:
<observation>
[what you can actually see and a reviewer could check: subjects, composition,
texture, colour, lighting, edges — facts about the picture itself. No verdict,
no guess about how it was made.]
</observation>
<reasoning>
[what those observations suggest, and what remains uncertain]
</reasoning>
<verdict>
{"verdict": "Real"|"Fake"|"Uncertain", "confidence": 0.0-1.0, "primary_evidence": [], "report": "concise report in Chinese"}
</verdict>
(confidence = your own uncalibrated certainty in THIS verdict, based on evidence
quality, expert applicability and unresolved questions; not P(Real/Fake).)
"""



def _load_image(path: str) -> Optional[Image.Image]:
    """Load an image; relative paths resolve against the project root."""
    try:
        resolved = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
        return Image.open(resolved).convert("RGB")
    except Exception:
        return None


def build_messages(image_path: str, history: List[Dict[str, str]],
                   system_prompt: str = FORENSIC_SYSTEM_PROMPT) -> List[dict]:
    """
    Convert internal conversation history into Qwen2.5-VL chat messages.

    Args:
        image_path: Path of the original image under analysis.
        history: List of {"from": "user"|"gpt", "value": str, ...} turns.
                 User turns may carry optional "image_paths" (region crops).
        system_prompt: System prompt variant (forensic default or the no-tool
                 baseline prompt used by the G2-d comparison).

    Returns:
        List of Qwen-style message dicts (role/content blocks).
    """
    messages: List[dict] = [
        {"role": "system", "content": system_prompt},
    ]

    for turn in history:
        role = turn.get("from", "user")
        value = turn.get("value", "")

        if role != "user":
            messages.append({"role": "assistant", "content": value})
            continue

        content_blocks: List[dict] = []

        if "<image>" in value:
            # Initial prompt: attach the original image.
            image = _load_image(image_path)
            if image is not None:
                content_blocks.append({"type": "image", "image": image})
            text = value.replace("<image>\n", "").replace("<image>", "")
        else:
            text = value
            # Diagnostic region crops attached to this turn (G1).
            for region_path in list(turn.get("image_paths") or [])[:MAX_REGION_IMAGES_PER_TURN]:
                image = _load_image(region_path)
                if image is not None:
                    content_blocks.append({"type": "image", "image": image})

        content_blocks.append({"type": "text", "text": text})
        messages.append({"role": "user", "content": content_blocks})

    if not history:
        # First turn: original image + task instruction.
        content_blocks: List[dict] = []
        image = _load_image(image_path)
        if image is not None:
            content_blocks.append({"type": "image", "image": image})
        content_blocks.append({"type": "text", "text": (
            "请分析这张图像的真实性，并使用法证工具箱开展多轮质证。"
            "首先输出 <planning> 标签，然后根据需要调用法证专家。"
        )})
        messages.append({"role": "user", "content": content_blocks})

    return messages


def collect_images(messages: List[dict]) -> List[Image.Image]:
    """Extract PIL images from message content blocks, in placeholder order."""
    images: List[Image.Image] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image":
                    image = block.get("image")
                    if isinstance(image, Image.Image):
                        images.append(image)
    return images
