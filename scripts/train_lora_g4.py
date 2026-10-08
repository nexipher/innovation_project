#!/usr/bin/env python3
"""
G4-g: LoRA fine-tuning on the admitted trajectories.

Trains on the *structured answer* of each sample, not on the whole session: the
tool-call turns are context (they are the base model's own calls, kept so the
conversation matches inference), and the behaviour being taught is the
four-section answer the policy stands behind.

Class balance is handled by the sampler rather than by discarding data — the
accepted set is Fake-heavy by construction (evidence only pays where the
no-tool baseline is confidently wrong), and training the raw mix would teach a
"tools mean Fake" prior.

The dataset construction, the message assembly and the weights are plain CPU
code and are tested without a model.  The training itself needs the GPU
approval from agent.md §3.2.

Usage:
  python scripts/train_lora_g4.py --dry-run                    # CPU: data only
  python scripts/train_lora_g4.py --run-name trial1            # GPU (authorized)
  python scripts/train_lora_g4.py --from-final-v2 --run-name provisional1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from mllm.message_builder import BASELINE_SYSTEM_PROMPT, build_forensic_prompt

REVIEWED_DIR = os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2_reviewed")
FINAL_V2_DIR = os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2")
RUNS_DIR = os.path.join(PROJECT_ROOT, "sft_data", "lora_runs")

REVIEWED_FILES = ("sft_accepted.json",)
PROVISIONAL_FILES = ("sft_tool_positive.json", "sft_no_tool_positive.json",
                     "sft_honest_abstention.json")

# LoRA on the language tower only: the vision tower is frozen, which is the
# usual choice when the task is "read the evidence correctly", not "see better".
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
DEFAULT_RANK = 16
DEFAULT_ALPHA = 32
DEFAULT_EPOCHS = 2
DEFAULT_LR = 1e-4


# ---------------------------------------------------------------------------
# Data (CPU-testable)
# ---------------------------------------------------------------------------

def load_training_samples(reviewed_dir: str = REVIEWED_DIR,
                          from_final_v2: bool = False,
                          final_v2_dir: str = FINAL_V2_DIR) -> List[dict]:
    """
    The samples to train on.

    Default source is the reviewed set — the reviewed one only exists once a
    human has been through it.  `from_final_v2` reads the validated but
    unreviewed set, which is what a provisional trial uses to find out whether
    the data and the objective work at all before asking anyone to review it.
    """
    directory = final_v2_dir if from_final_v2 else reviewed_dir
    names = PROVISIONAL_FILES if from_final_v2 else REVIEWED_FILES
    samples: List[dict] = []
    for name in names:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            samples.extend(json.load(handle))
    return samples


def system_prompt_for(sample: dict) -> str:
    """
    The prompt this sample's conversation was produced under.

    A tool-free sample was run with the baseline prompt, a tool sample with the
    forensic prompt listing exactly the tools it was allowed to call — training
    it under a different prompt would teach it to answer a question it is not
    being asked at inference.
    """
    tools = (sample.get("metadata") or {}).get("tools_served") or []
    if not tools:
        return BASELINE_SYSTEM_PROMPT
    # `tools_served` already holds tool keys ("freq"/"noise"/"jpeg").
    return build_forensic_prompt(tools)


def split_answer(sample: dict) -> Tuple[List[dict], dict]:
    """
    Separate the conversation into (prompt turns, answer turn).

    The answer is the last assistant turn; everything before it — the task, the
    tool calls, the evidence messages — is context that must be present but
    must not be trained on.
    """
    conversations = sample.get("conversations") or []
    if not conversations or conversations[-1].get("from") != "gpt":
        raise ValueError(f"{sample.get('id')}: conversation does not end with the answer")
    return conversations[:-1], conversations[-1]


def sampling_weights(samples: List[dict], balance: bool = True) -> Tuple[List[float], dict]:
    """
    Per-sample weights that make one epoch see the classes equally.

    Returned rather than applied by dropping samples: the Fake-heavy
    composition is informative, and discarding it would also discard the
    trajectories that show evidence working.
    """
    labels = Counter(s.get("ground_truth") for s in samples)
    if not balance or not labels:
        return [1.0] * len(samples), dict(labels)
    largest = max(labels.values())
    per_label = {label: largest / count for label, count in labels.items()}
    return [per_label[s.get("ground_truth")] for s in samples], dict(labels)


def summarise(samples: List[dict], weights: List[float]) -> dict:
    """What the run is about to train on, for the manifest and the dry run."""
    tools = Counter(",".join(s.get("metadata", {}).get("tools_served") or []) or "no-tool"
                    for s in samples)
    return {
        "samples": len(samples),
        "labels": dict(Counter(s.get("ground_truth") for s in samples)),
        "buckets": dict(Counter(s.get("type") for s in samples)),
        "treatments": dict(Counter((s.get("metadata") or {}).get("treatment") for s in samples)),
        "policies": dict(tools),
        "mean_weight": round(sum(weights) / len(weights), 4) if weights else None,
        "max_weight": round(max(weights), 4) if weights else None,
    }


def build_messages(sample: dict, load_images: bool = False):
    """
    (prompt_messages, full_messages, images) for loss masking.

    The prompt edition stops before the answer, so tokenising both gives the
    boundary to mask at.  With `load_images` the image blocks carry PIL images,
    which is what the processor needs; without it they carry paths, which is
    what the CPU tests assert on.
    """
    from PIL import Image

    prompt_turns, answer = split_answer(sample)
    system = system_prompt_for(sample)
    images: List = []

    def image_block(path: str) -> dict:
        absolute = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
        if load_images:
            images.append(Image.open(absolute).convert("RGB"))
            return {"type": "image", "image": images[-1]}
        return {"type": "image", "path": path}

    def to_messages(turns):
        messages = [{"role": "system", "content": system}]
        for index, turn in enumerate(turns):
            if turn.get("from") != "user":
                messages.append({"role": "assistant", "content": turn.get("value", "")})
                continue
            value = turn.get("value", "")
            content = []
            # The opening turn carries the `<image>` marker; resolving it here
            # is what the message builder does at inference, and without it the
            # model would be trained to judge an image it never saw.
            if index == 0 and "<image>" in value:
                content.append(image_block(sample["image_path"]))
                value = value.replace("<image>\n", "").replace("<image>", "")
            content.extend(image_block(path) for path in turn.get("image_paths") or [])
            content.append({"type": "text", "text": value})
            messages.append({"role": "user", "content": content})
        return messages

    prompt_messages = to_messages(prompt_turns)
    full_messages = prompt_messages + [{"role": "assistant", "content": answer["value"]}]
    return prompt_messages, full_messages, images


# ---------------------------------------------------------------------------
# Dataset (needs the processor)
# ---------------------------------------------------------------------------

class TrajectoryDataset:
    """
    One tokenised sample per item, prompt masked out of the loss.

    Batch size stays 1 on purpose: Qwen-VL builds vision tensors per image, and
    batching would mean padding text and stitching image grids for no benefit —
    gradient accumulation gives the effective batch size instead.
    """

    def __init__(self, samples: List[dict], processor, max_length: int = 4096):
        self._samples = samples
        self._processor = processor
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._samples)

    def encode(self, messages: List[dict], images: List) -> dict:
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False)
        kwargs = {"text": [text], "return_tensors": "pt", "padding": True}
        if images:
            kwargs["images"] = images
        return self._processor(**kwargs)

    def __getitem__(self, index: int) -> dict:
        sample = self._samples[index]
        prompt_messages, full_messages, images = build_messages(sample, load_images=True)

        full = self.encode(full_messages, images)
        prompt = self.encode(prompt_messages, images)

        input_ids = full["input_ids"][0]
        labels = input_ids.clone()
        prompt_length = min(prompt["input_ids"].shape[1], labels.shape[0])
        labels[:prompt_length] = -100

        # Keep every tensor exactly as the processor returned it.  `input_ids`
        # must stay 2-D — the vision position computation indexes shape[1] —
        # while `pixel_values` (patches, dim) and `image_grid_thw` (images, 3)
        # carry no batch dimension, and indexing those with [0] would silently
        # train on the first image only.
        item = {key: value for key, value in full.items() if key != "input_ids"}
        item["input_ids"] = input_ids.unsqueeze(0)
        item["labels"] = labels.unsqueeze(0)
        item["sample_id"] = sample["id"]
        item["prompt_tokens"] = int(prompt_length)
        item["answer_tokens"] = int(labels.shape[0] - prompt_length)
        return item

    @property
    def samples(self) -> List[dict]:
        return self._samples


# ---------------------------------------------------------------------------
# Training (GPU)
# ---------------------------------------------------------------------------

def train(samples: List[dict], run_name: str, epochs: int, lr: float,
          rank: int, alpha: int, max_length: int) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    from config import QWEN_MODEL_PATH

    print(f"[train] loading {QWEN_MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_PATH, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    # Enable checkpointing on the base model *before* wrapping: peft proxies
    # the call, but the decoder modules have to see the flag when the graph is
    # built, and a wrapped call is easy to get wrong silently.
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    print(f"[train] gradient checkpointing on base: {model.is_gradient_checkpointing}")

    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=list(LORA_TARGETS),
    )
    model = get_peft_model(model, config)
    # After wrapping: without `enable_input_require_grads` a checkpointed graph
    # gives the adapters no gradient at all, and with the reentrant
    # implementation the recomputation does not happen — the activations stay
    # alive instead, which is what exhausted the card.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    print(f"[train] gradient checkpointing after wrap: {model.is_gradient_checkpointing}")
    _freeze_vision_backward(model)
    model.print_trainable_parameters()

    dataset = TrajectoryDataset(samples, processor, max_length)
    weights, _ = sampling_weights(samples)
    sampler = torch.utils.data.WeightedRandomSampler(
        weights, num_samples=len(dataset), replacement=True)

    def collate(batch):
        # Batch size 1: the item already carries exactly what the model needs.
        item = dict(batch[0])
        item.pop("sample_id", None)
        item.pop("prompt_tokens", None)
        item.pop("answer_tokens", None)
        return item

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, sampler=sampler, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr)

    output_dir = os.path.join(RUNS_DIR, run_name)
    os.makedirs(output_dir, exist_ok=True)
    model.train()
    losses = []
    for epoch in range(epochs):
        for step, batch in enumerate(loader):
            batch = {k: v.to(model.device) if hasattr(v, "to") else v
                     for k, v in batch.items()}
            loss = compute_masked_loss(model, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(float(loss))
            if step % 10 == 0:
                print(f"[epoch {epoch}] step {step}/{len(loader)} loss {losses[-1]:.4f}")

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    print(f"[train] adapter saved to {output_dir}")
    return {"loss_first": losses[0], "loss_last": losses[-1],
            "steps": len(losses), "output_dir": output_dir}


def compute_masked_loss(model, batch: dict):
    """
    Cross-entropy on the supervised positions only.

    The model's own loss projects every position through a 152k-token head —
    742 MB of logits and as much again in gradients, for a sample whose answer
    is a fifth of the sequence.  Running the head on the masked positions keeps
    the same objective at a fraction of the memory.
    """
    import torch
    import torch.nn.functional as F

    labels = batch["labels"]
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        pixel_values=batch.get("pixel_values"),
        image_grid_thw=batch.get("image_grid_thw"),
        output_hidden_states=True,
        use_cache=False,
    )
    hidden = outputs.hidden_states[-1][:, :-1, :]
    shifted = labels[:, 1:]
    supervised = shifted != -100
    logits = model.lm_head(hidden[supervised])
    return F.cross_entropy(logits.float(), shifted[supervised])


def _freeze_vision_backward(model) -> None:
    """
    Keep the vision tower out of the backward graph.

    It is frozen — LoRA trains the language tower — but a frozen module still
    stores activations for the backward pass, and for a multi-image sample that
    alone exhausted a 24 GB card.  Running it under `no_grad` makes its output
    a constant; the graph then starts at the language embeddings, which
    `enable_input_require_grads` has already hooked.
    """
    import torch

    # Wrapping (PeftModel -> LoraModel -> the VL model -> its vision tower)
    # makes the attribute path brittle, so find it by class name instead.
    visual = None
    for name, module in model.named_modules():
        class_name = type(module).__name__.lower()
        if "vision" in class_name and hasattr(module, "forward"):
            visual = module
            break
    if visual is None:
        print("[train] WARNING: no vision tower found to detach; "
              "the backward pass may not fit")
        return

    original_forward = visual.forward

    def forward_without_grad(*args, **kwargs):
        with torch.no_grad():
            return original_forward(*args, **kwargs)

    visual.forward = forward_without_grad
    print("[train] vision tower runs outside the backward graph")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-dir", default=REVIEWED_DIR)
    parser.add_argument("--final-v2-dir", default=FINAL_V2_DIR)
    parser.add_argument("--from-final-v2", action="store_true",
                        help="train on the validated-but-unreviewed set (provisional trial)")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--rank", type=int, default=DEFAULT_RANK)
    parser.add_argument("--alpha", type=int, default=DEFAULT_ALPHA)
    parser.add_argument("--max-length", type=int, default=2560)
    parser.add_argument("--dry-run", action="store_true",
                        help="CPU: build and check the data, load no model")
    args = parser.parse_args()

    samples = load_training_samples(args.reviewed_dir, args.from_final_v2,
                                    args.final_v2_dir)
    if not samples:
        raise SystemExit(
            "no samples to train on: the reviewed set only exists after G4-f; "
            "use --from-final-v2 for a provisional trial")
    weights, _ = sampling_weights(samples)
    summary = summarise(samples, weights)
    print(f"samples: {summary['samples']}  labels: {summary['labels']}  "
          f"buckets: {summary['buckets']}")
    print(f"policies: {summary['policies']}")
    print(f"sampling weights: mean {summary['mean_weight']} max {summary['max_weight']}")

    # Every sample must assemble into a trainable conversation before any GPU
    # time is spent on it.
    for sample in samples:
        prompt_messages, full_messages, _ = build_messages(sample)
        assert full_messages[-1]["role"] == "assistant", sample["id"]
        assert len(full_messages) == len(prompt_messages) + 1, sample["id"]
    print(f"message assembly: ok for all {len(samples)} samples")

    if args.dry_run:
        print("dry run: no model loaded, no GPU used")
        return

    if not _cuda_available():
        raise SystemExit(
            "CUDA not available. Training requires the RTX 4090 (agent.md §3.2: "
            "obtain user authorization first). Use --dry-run on CPU.")

    run_name = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    result = train(samples, run_name, args.epochs, args.lr, args.rank,
                   args.alpha, args.max_length)

    from utils import config_fingerprint

    manifest = {
        "run_name": run_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": "final_v2_provisional" if args.from_final_v2 else "reviewed",
        "lora": {"rank": args.rank, "alpha": args.alpha, "targets": list(LORA_TARGETS)},
        "epochs": args.epochs, "lr": args.lr, "max_length": args.max_length,
        "data": summary,
        "result": {k: v for k, v in result.items() if k != "output_dir"},
        "config_fingerprint": config_fingerprint.compute(_default_experts(),
                                                        result["output_dir"]),
    }
    with open(os.path.join(result["output_dir"], "run_manifest.json"), "w",
              encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"manifest: {os.path.join(result['output_dir'], 'run_manifest.json')}")


def _default_experts() -> dict:
    from experts.frequency_v2 import FrequencyExpertV2
    from experts.jpeg import JPEGExpert
    from experts.noise import NoiseExpert

    return {"frequency_expert_v2": FrequencyExpertV2(),
            "noise_expert": NoiseExpert(), "jpeg_expert": JPEGExpert()}


if __name__ == "__main__":
    main()
