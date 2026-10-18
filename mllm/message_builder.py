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
}

TOOL_MEASURES: Dict[str, str] = {
    "probe": "- <call_probe>  AI-generation discriminator over the whole image. STRONGEST evidence\n  available (separation ≈0.98 on its fitting group, holds on generators it never saw).\n  Its score IS P(AI-generated): HIGH points to Fake, LOW points to Real. It measures the\n  image as a whole and locates NOTHING — never restate it as a physical artifact or as\n  an area of the image.",
    "freq": "- <call_freq>  multi-scale frequency analysis. WEAK evidence (separation 0.56-0.63,\n  barely above chance). Use it to corroborate, never as the deciding signal.",
    "noise": "- <call_noise> residual micro-noise LEVEL. COUNTER-INTUITIVE: a HIGH level points to\n  Real (camera sensor micro-noise) and a LOW level points to Fake (smooth generator\n  output). This is stable across image formats. Do NOT read a high value as forgery.",
    "jpeg": "- <call_jpeg>  JPEG compression history (blockiness / DCT structure). A HIGH level\n  points to Real (a camera JPEG that was saved or re-saved), NOT to forgery. After\n  heavy re-compression (quality <= 70) this measurement is near chance and must be\n  ignored.",
}

# 登记表支持的工具有序表；freq/jpeg/noise 是待退役的旧工具，保留其手写文本供
# 历史复算与 v2 对照装配提示词（阶段 4.4：提示词由登记表生成，不再手写新工具）。
TOOL_ORDER = ("probe", "npr", "univfd", "noise", "freq", "jpeg")
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
    "(confidence = your own uncalibrated certainty in THIS verdict, not P(Real/Fake): RULES 9)\n"
    "</verdict>\n\n")

PROMPT_TAIL_BASE = (
    "\nEach evidence token you receive has EXACTLY these fields:\n"
    "  `evidence_id`, `expert`, `measurement`, `status`, `scope`,\n"
    "  `score` = {`name`, `value`, `range`, `direction`, `direction_for_this_value`,\n"
    "             `threshold`, `threshold_chosen_on`,\n"
    "             `is_calibrated_probability`: always false},\n"
    "  `reading` = a short account of what was measured and the conditions under which\n"
    "              the number may be read,\n"
    "  and, when the expert rendered one, `visual_artifacts` + `artifact_meaning`.\n"
    "\n1. There is ONE score per token. Its `direction` says which side means what —\n"
    "   follow it, never your own expectation about the expert. Scores from different\n"
    "   experts are NOT comparable and must not be averaged or ranked against each other.\n"
    "2. Read the `reading` before using the number: it states the conditions. `direction`\n"
    "   is the expert's scale; `direction_for_this_value` is what THIS number means under\n"
    "   those conditions. When it is `Uncertain`, the number carries no direction here:\n"
    "   say so instead of forcing one.\n"
    "3. The number is not a probability, even when it lies between 0 and 1.\n"
    "4. `status` tells you whether a measurement happened: a `failed` or missing\n"
    "   measurement is NOT a neutral result, and must never be reported as one.\n"
    "5. `scope` says what the expert could see: `global` = the whole image;\n"
    "   `center_crop` = it never saw the edges. Do not restate a crop-based measurement\n"
    "   as a whole-image finding, and do not restate a whole-image score as describing\n"
    "   an area.\n"
    "6. `visual_artifacts` are diagnostic products for human review, not your evidence:\n"
    "   the number above is.\n"
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

PROMPT_TAIL_RULES = "\nRULES:\n1. Call at most one more expert than you need: prefer the expert whose measurement is\n   most likely to discriminate the specific anomaly you described in <planning>.\n2. Call each expert AT MOST ONCE per session. Every expert measures the WHOLE image,\n   so a second call — whatever region you name — returns the same measurement: the\n   system answers it with the numbers you already have, and the turn is wasted.\n   This is a hard rule, not advice.\n3. EVERY response carries exactly one action tag: one <call_x> (bbox optional) or\n   one <verdict>. A response with no tag, or with more than one action, is refused.\n4. NEVER fabricate evidence — only reference evidence tokens you have received.\n5. You decide when the evidence is sufficient to conclude. Conclude once the measurements\n   you called for have arrived and another call would not change your reading; the system\'s\n   budget — not a count of tokens — is what ends a session that never concludes. If the\n   evidence is weak, conflicting, or mostly `Uncertain`/`disabled`, say \"Uncertain\" rather\n   than guessing: an honest Uncertain is preferred over a confident mistake.\n6. Describe only what you can SEE before a call — appearance, texture, edges, lighting.\n   You cannot see SRM residuals, DCT histograms or sensor noise in the picture itself;\n   make those claims only after the expert that measures them has returned a token.\n7. Every conclusion states the scope of the evidence: how many distinct experts returned\n   a VALID measurement, and whether you compared them ACROSS EXPERTS. Call it a cross-expert\n   comparison, never \"independent cross-validation\": two detectors can share training\n   sources or fail together. One measurement is not a comparison, and calling a second\n   expert does not by itself make the verification independent. State what the missing\n   evidence limits only when it bears on the conclusion.\n8. Visible artefacts are observations (blockiness, blur, banding): describe them, do not\n   assert their cause. Never claim that compression, re-encoding or sensor noise has been\n   measured, ruled out or shown not to affect a result — only the expert that measures\n   them may say anything about them, and only within its recorded `scope`. Take image\n   dimensions from the input metadata; never infer \"high resolution\" from appearance.\n9. `confidence` is YOUR OWN subjective, self-reported certainty in the verdict you chose.\n   It is NOT a probability that the image is real or AI-generated, not calibrated, and not\n   an accuracy. Never build it by copying, inverting, averaging or multiplying expert\n   scores. Keep three things apart: the expert\'s raw score, your self-reported confidence,\n   and the project\'s measured performance. When the verdict is \"Uncertain\", `confidence`\n   is your certainty about abstaining — never P(Fake). No per-report disclaimers are\n   needed: this contract, the field definitions and the review view carry the semantics.\n"

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
        tools = [t for t in TOOL_ORDER if t in set(allowed_tools)]
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
    if len(tools) < len(TOOL_ORDER):
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

    return (PROMPT_HEAD
            + "AVAILABLE ACTIONS (use EXACTLY these tag names — do NOT invent variations):\n"
            + actions + "\n\nFORBIDDEN:" + PROMPT_MID
            + measures + "HOW TO READ AN EVIDENCE TOKEN:"
            + PROMPT_TAIL_BASE + read_notes + PROMPT_TAIL_RULES)


FORENSIC_SYSTEM_PROMPT = build_forensic_prompt()

MAX_REGION_IMAGES_PER_TURN = 2

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
(confidence = your own uncalibrated certainty in THIS verdict, not P(Real/Fake).)
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
