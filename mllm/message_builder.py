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
    "freq": "- <call_freq>[ymin, xmin, ymax, xmax]</call_freq>  → call frequency-domain expert",
    "noise": "- <call_noise>[ymin, xmin, ymax, xmax]</call_noise> → call noise residual expert",
    "jpeg": "- <call_jpeg>[ymin, xmin, ymax, xmax]</call_jpeg>   → call JPEG compression expert",
}

TOOL_MEASURES: Dict[str, str] = {
    "freq": "- <call_freq>  multi-scale frequency analysis. WEAK evidence (separation 0.56-0.63,\n  barely above chance). Use it to corroborate, never as the deciding signal.",
    "noise": "- <call_noise> residual micro-noise LEVEL. COUNTER-INTUITIVE: a HIGH level points to\n  Real (camera sensor micro-noise) and a LOW level points to Fake (smooth generator\n  output). This is stable across image formats. Do NOT read a high value as forgery.",
    "jpeg": "- <call_jpeg>  JPEG compression history (blockiness / DCT structure). A HIGH level\n  points to Real (a camera JPEG that was saved or re-saved), NOT to forgery. After\n  heavy re-compression (quality <= 70) this measurement is near chance and must be\n  ignored.",
}

TOOL_ORDER = ("freq", "noise", "jpeg")

PROMPT_HEAD = "You are an AI forensic image analyst. You MUST follow this EXACT format in EVERY response. Do NOT write free-form analysis.\n\n"
PROMPT_MID = " Do NOT use <call_call_freq>, <call_call_noise>, <call_frequency>, or any other variation.\n\nCOORDINATES: bbox values are integers in range [0, 1000], format [ymin, xmin, ymax, xmax].\n\nRESPONSE FORMAT (MANDATORY — every response must contain one of these two structures):\n\nStructure A — When you need forensic evidence:\n<planning>\nSuspected Region: [ymin, xmin, ymax, xmax]\nVisual Anomalies: [describe what looks suspicious in this specific image]\nExpert Target & Hypothesis: [which expert to call and why]\n</planning>\n<call_EXPERT_NAME>[ymin, xmin, ymax, xmax]</call_EXPERT_NAME>\n\nStructure B — When you have enough evidence to conclude:\n<reasoning>\n[Cross-reference the expert's physical findings with your visual observations.\nIf different experts conflict, explain why and apply \"presumption of innocence\".\nIf the image has compression artifacts that may weaken certain signals, note it.]\n</reasoning>\n<verdict>\n{\"verdict\": \"Real\"|\"Fake\"|\"Uncertain\", \"confidence\": 0.0-1.0, \"primary_evidence\": [\"evidence_name1\"], \"report\": \"concise forensic report in Chinese\"}\n</verdict>\n\n"
PROMPT_TAIL = "\n1. `calibrated_likelihood` is the authoritative direction — it is the empirical\n   P(Real)/P(Fake) for that metric band. `strength` is NOT a probability and NOT\n   comparable between experts (each has its own scale); never rank experts by it.\n2. Check `applicability` and `applicability_conditions` before using a token. A token\n   labelled `disabled:*` must not influence the verdict; `weak:*` may only corroborate.\n3. `counter_explanation` lists the benign causes of the same phenomenon — if it also\n   explains what you see, do not treat the evidence as incriminating.\n4. Weight the evidence by `reliability`; conflicting tokens cancel out.\n5. `measurement_scope: \"global\"` means the expert measured the WHOLE image. The bbox\n   you provide locates the diagnostic region — the crop and the artifacts you receive\n   — it does not restrict the measurement. Judge the whole image, not the crop.\n\nRULES:\n1. Call at most one more expert than you need: prefer the expert whose measurement is\n   most likely to discriminate the specific anomaly you described in <planning>.\n2. Never call an expert you have already called in this session: each one measures\n   the WHOLE image, so a second call — whatever region you name — returns the same\n   measurement, which is suppressed and wastes the budget. Prefer an expert you have\n   not used yet.\n3. NEVER output only natural-language analysis without the required XML tags.\n4. NEVER fabricate evidence — only reference evidence tokens you have received.\n5. After receiving 2+ evidence tokens, you MUST produce a verdict. If the evidence is\n   weak, conflicting, or mostly `Uncertain`/`disabled`, output \"Uncertain\" rather than\n   guessing: an honest Uncertain is preferred over a confident mistake.\n"

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

    actions = "\n".join(TOOL_ACTIONS[t] for t in tools)
    if len(tools) < len(TOOL_ORDER):
        actions = actions + "\n" + SUBSET_NOTE
    measures = "\n".join(TOOL_MEASURES[t] for t in tools)

    return (PROMPT_HEAD
            + "AVAILABLE ACTIONS (use EXACTLY these tag names — do NOT invent variations):\n"
            + actions + "\n\nFORBIDDEN:" + PROMPT_MID
            + "WHAT EACH TOOL ACTUALLY MEASURES (G2-b calibration, format-balanced set of "
              "700 samples):\n" + measures + "\n\nHOW TO READ AN EVIDENCE TOKEN:"
            + PROMPT_TAIL)


FORENSIC_SYSTEM_PROMPT = build_forensic_prompt()

MAX_REGION_IMAGES_PER_TURN = 2

# No-tool variant (G2-d condition 1): the RGB baseline must answer from the
# image alone, so tool calls are forbidden instead of merely unavailable.
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
