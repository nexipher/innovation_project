"""Qwen3-VL LoRA smoke on three approved SFT candidate trajectories.

Training data are the *sessions* of:
  * autosplice__55310__1__q95            (newspaper; TruFor then PROBE)
  * autosplice__65547__authentic__q95    (shoulder; TruFor then PROBE)
  * single_trufor__autosplice__54029__1__q95 (TruFor only)

The original image is attached to the first user turn; the three TruFor panels
are attached to the evidence turn that follows the TruFor call; the PROBE crop
follows the PROBE call.  Only assistant turns are supervised: the call turns
and the closing turn are in the loss, while user/tool-result turns are context
(labels = -100).  This is the same loss-masking rule as
`scripts/train_lora_g4.py`, adapted to Qwen3-VL-8B-Instruct.

Usage:
  python scripts/train_lora_candidate_smoke.py --dry-run
  FORENSIC_ALLOW_GPU=1 python scripts/train_lora_candidate_smoke.py --max-steps 6
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

from config import PROJECT_ROOT  # noqa: E402
from mllm.message_builder import build_forensic_prompt  # noqa: E402
from mllm.qwen3_client import MODEL_ID, MODEL_REVISION  # noqa: E402
from scripts.train_lora_g4 import assistant_spans  # noqa: E402

OUT_DIR = ROOT / "sft_data/lora_smoke_20261128"
CANDIDATE_DIR = ROOT / "sft_data/sft_candidates_20261127"
SAMPLE_IDS = (
    "autosplice__55310__1__q95",
    "autosplice__65547__authentic__q95",
    "single_trufor__autosplice__54029__1__q95",
)
SYSTEM_PROMPT = build_forensic_prompt(("probe", "trufor"))
# Attention-only: MLP adapters add a [tokens, 11008] branch per layer and made
# the 9k-token multimodal smoke OOM on the second step of a 24 GB card.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def _load_samples() -> list[dict]:
    records = {row["sample_id"]: row for row in json.loads(
        (CANDIDATE_DIR / "records.json").read_text(encoding="utf-8"))}
    samples = []
    for sid in SAMPLE_IDS:
        record = records[sid]
        trace = json.loads((ROOT / record["trace_path"]).read_text(encoding="utf-8"))
        samples.append({
            "id": sid,
            "image_path": record["image_path"],
            "source_image": record["source_image"],
            "ground_truth": record["ground_truth"],
            "conversations": trace["conversations"],
            "metadata": {"tools_served": ["probe", "trufor"]},
        })
    return samples


def build_messages(sample: dict):
    from PIL import Image

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    images = []
    for index, turn in enumerate(sample["conversations"]):
        value = turn.get("value", "")
        if turn.get("from") != "user":
            messages.append({"role": "assistant", "content": value})
            continue
        content = []
        if index == 0 and "<image>" in value:
            image = Image.open(ROOT / sample["image_path"]).convert("RGB")
            images.append(image)
            content.append({"type": "image", "image": image})
            value = value.replace("<image>\n", "").replace("<image>", "")
        for path in turn.get("image_paths") or []:
            image = Image.open(ROOT / path).convert("RGB")
            images.append(image)
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": value})
        messages.append({"role": "user", "content": content})
    return messages, images


def validate_sample(sample: dict, processor=None) -> dict:
    messages, images = build_messages(sample)
    gpt_turns = [turn for turn in sample["conversations"] if turn.get("from") == "gpt"]
    user_turns = [turn for turn in sample["conversations"] if turn.get("from") == "user"]
    tool_turns = [turn for turn in user_turns if turn.get("image_paths")]
    if not messages or messages[-1]["role"] != "assistant":
        raise SystemExit(f"{sample['id']}: no closing assistant turn")

    first_user = next((message for message in messages
                       if message.get("role") == "user"), None)
    first = first_user["content"] if first_user else []
    if not any(block.get("type") == "image" for block in first):
        raise SystemExit(f"{sample['id']}: original image not attached to first user turn")
    if len(images) != 1 + sum(len(turn.get("image_paths") or []) for turn in user_turns):
        raise SystemExit(f"{sample['id']}: image count mismatch")

    panel_names = []
    for turn in tool_turns:
        names = [os.path.basename(p) for p in turn.get("image_paths") or []]
        panel_names.append(names)
    if len(tool_turns) >= 1:
        first_names = panel_names[0]
        for required in ("model_original_crop", "model_anomaly_map",
                         "model_confidence_map"):
            if not any(required in name for name in first_names):
                raise SystemExit(f"{sample['id']}: first tool turn lacks {required}")
    if len(tool_turns) >= 2 and not any("region_" in name for name in panel_names[1]):
        raise SystemExit(f"{sample['id']}: second tool turn lacks PROBE crop")
    if len(tool_turns) == 1 and len(gpt_turns) == 2:
        # Single TruFor sample: one tool turn with three panels, no PROBE crop.
        if len(panel_names[0]) != 3:
            raise SystemExit(f"{sample['id']}: expected exactly three TruFor panels")

    info = {
        "sample_id": sample["id"],
        "assistant_turns": len(gpt_turns),
        "tool_turns": len(tool_turns),
        "images_per_tool_turn": [len(names) for names in panel_names],
        "total_pil_images": len(images),
        "calls": ["trufor" if any("model_original_crop" in n for n in names)
                  else "probe" for names in panel_names],
    }
    if processor is None:
        return info

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
        enable_thinking=False)
    encoded = processor(text=[text], images=images, return_tensors="pt",
                        padding=True)
    input_ids = encoded["input_ids"][0]
    tokenizer = getattr(processor, "tokenizer", processor)
    spans = assistant_spans(input_ids.tolist(), tokenizer)
    if len(spans) != len(gpt_turns):
        raise SystemExit(
            f"{sample['id']}: assistant spans {len(spans)} != gpt turns {len(gpt_turns)}")
    supervised = sum(stop - start for start, stop in spans)
    info.update({
        "input_tokens": int(input_ids.numel()),
        "supervised_turns": len(spans),
        "supervised_tokens": supervised,
        "masked_tokens": int(input_ids.numel()) - supervised,
    })
    return info


def _snapshot() -> str:
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL_ID, revision=MODEL_REVISION,
                             local_files_only=True)


def _freeze_vision_backward(model) -> None:
    import torch
    visual = None
    for _name, module in model.named_modules():
        if "vision" in type(module).__name__.lower() and hasattr(module, "forward"):
            visual = module
            break
    if visual is None:
        print("[train] WARNING: no vision tower found")
        return
    original = visual.forward

    def forward_without_grad(*args, **kwargs):
        with torch.no_grad():
            return original(*args, **kwargs)

    visual.forward = forward_without_grad
    print("[train] vision tower runs outside the backward graph")


def _compute_loss(model, item: dict):
    import torch
    import torch.nn.functional as F

    labels = item["labels"]
    # Call the decoder directly for the last hidden state.  Requesting
    # `output_hidden_states=True` on the 36-layer 9k-token multimodal sample
    # retains ~2.6 GB of intermediate states and OOMs a 24 GB card on the
    # second step.  The top-level forward only adds the LM head, which we apply
    # to the supervised positions below.
    base = model.base_model.model if hasattr(model, "base_model") else model
    outputs = base.model(
        input_ids=item["input_ids"],
        attention_mask=item.get("attention_mask"),
        mm_token_type_ids=item.get("mm_token_type_ids"),
        pixel_values=item.get("pixel_values"),
        image_grid_thw=item.get("image_grid_thw"),
        use_cache=False,
    )
    hidden = outputs[0] if isinstance(outputs, tuple) else outputs.last_hidden_state
    hidden = hidden[:, :-1, :]
    shifted = labels[:, 1:]
    supervised = shifted != -100
    lm_head = getattr(base, "lm_head", None)
    if lm_head is None:
        lm_head = model.get_output_embeddings()
    logits = lm_head(hidden[supervised])
    return F.cross_entropy(logits.float(), shifted[supervised])


def train(samples: list[dict], steps: int, lr: float, rank: int, alpha: int,
          max_length: int, out: Path, optimizer_name: str = "adamw") -> dict:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

    snapshot = _snapshot()
    print(f"[train] model {MODEL_ID}@{MODEL_REVISION[:8]} from {snapshot}")
    processor = Qwen3VLProcessor.from_pretrained(snapshot, local_files_only=True)
    infos = [validate_sample(sample, processor) for sample in samples]
    for info in infos:
        if info["input_tokens"] > max_length:
            raise SystemExit(
                f"{info['sample_id']}: {info['input_tokens']} tokens > max_length "
                f"{max_length}; refusing silent truncation")
    print("[data] " + json.dumps(infos, ensure_ascii=False))

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.bfloat16, device_map="auto",
        local_files_only=True)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGETS))
    model = get_peft_model(model, config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    _freeze_vision_backward(model)
    model.print_trainable_parameters()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name == "sgd":
        # The 24 GB card cannot hold AdamW's two float32 moments for the 43.6M
        # trainable parameters on top of the 9k-token multimodal activations.
        # SGD is stateless; this is a smoke, not a convergence run.
        optimizer = torch.optim.SGD(params, lr=lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=lr)

    encoded = []
    for sample in samples:
        messages, images = build_messages(sample)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
            enable_thinking=False)
        batch = processor(text=[text], images=images, return_tensors="pt",
                          padding=True)
        tokenizer = getattr(processor, "tokenizer", processor)
        spans = assistant_spans(batch["input_ids"][0].tolist(), tokenizer)
        labels = torch.full_like(batch["input_ids"], -100)
        for start, stop in spans:
            labels[0, start:stop] = batch["input_ids"][0, start:stop]
        batch["labels"] = labels
        encoded.append((sample["id"], {k: v for k, v in batch.items()}))

    torch.cuda.reset_peak_memory_stats()
    losses = []
    for step in range(steps):
        sid, batch = encoded[step % len(encoded)]
        batch = {k: (v.to(model.device) if hasattr(v, "to") else v)
                 for k, v in batch.items()}
        loss = _compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(round(float(loss.detach()), 6))
        print(f"[step {step + 1}/{steps}] {sid} loss {losses[-1]:.6f}")
        del batch
        gc.collect()

    peak_alloc = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    peak_reserved = round(torch.cuda.max_memory_reserved() / 1e9, 3)
    out.mkdir(parents=True, exist_ok=True)
    adapter = out / "adapter"
    model.save_pretrained(adapter)
    processor.save_pretrained(adapter)
    print(f"[train] adapter saved to {adapter}")
    return {
        "steps": steps, "losses": losses,
        "loss_first": losses[0], "loss_last": losses[-1],
        "peak_allocated_gb": peak_alloc, "peak_reserved_gb": peak_reserved,
        "adapter": str(adapter), "data": infos,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR))
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=10000)
    parser.add_argument("--optimizer", choices=("adamw", "sgd"), default="adamw")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    out = Path(args.out)
    samples = _load_samples()

    if args.dry_run:
        from transformers import Qwen3VLProcessor
        snapshot = _snapshot()
        processor = Qwen3VLProcessor.from_pretrained(snapshot, local_files_only=True)
        infos = [validate_sample(sample, processor) for sample in samples]
        print(json.dumps({"dry_run": True, "samples": infos}, ensure_ascii=False,
                         indent=2))
        return 0

    if os.environ.get("FORENSIC_ALLOW_GPU") != "1":
        raise SystemExit("GPU 未授权：设置 FORENSIC_ALLOW_GPU=1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    result = train(samples, args.max_steps, args.lr, args.rank, args.alpha,
                   args.max_length, out, optimizer_name=args.optimizer)
    manifest = {
        "schema": "lora_smoke/1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "data": {"sample_ids": list(SAMPLE_IDS),
                 "system_prompt_sha256": __import__("hashlib").sha256(
                     SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
                 "samples": result["data"]},
        "training": {"max_steps": result["steps"], "lr": args.lr,
                     "rank": args.rank, "alpha": args.alpha,
                     "max_length": args.max_length,
                     "optimizer": args.optimizer,
                     "target_modules": list(LORA_TARGETS),
                     "cuda_alloc_conf": os.environ.get(
                         "PYTORCH_CUDA_ALLOC_CONF", "")},
        "result": {k: v for k, v in result.items() if k != "data"},
    }
    (out / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    import torch
    raise SystemExit(main())
