#!/usr/bin/env python3
"""
SFT data generation at scale — Phase 2.3.

Runs Qwen2.5-VL on 600+ images (A-line: natural forensic pipeline)
and generates 300+ constructed scenarios (B-line: conflict/intervention).
Outputs ShareGPT-format JSON ready for Phase 3 SFT training.

Usage:
  python scripts/generate_sft_data.py --stream A --num-real 200 --num-fake 50
  python scripts/generate_sft_data.py --stream B --num-scenarios 300
  python scripts/generate_sft_data.py --stream both
  python scripts/generate_sft_data.py --stream A --dry-run   # check setup without GPU
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    REAL_DIR, GENIMAGE_TEST_DIR, GENIMAGE_SUBDIRS, SYSTEM_PROMPT,
)
from utils.image_utils import ImageUtils
from utils.parser import Parser
from utils.coordinate_transformer import CoordinateTransformer
from utils.logger import SessionLogger
from state_machine.controller import ForensicStateMachine
from state_machine.evidence_tokenizer import EvidenceTokenizer
from state_machine.halting import HaltingChecker
from experts.frequency import FrequencyExpert
from experts.noise import NoiseExpert
from experts.jpeg import JPEGExpert


# ---------------------------------------------------------------------------
# Output setup
# ---------------------------------------------------------------------------
SFT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sft_data"
)
STREAM_A_DIR = os.path.join(SFT_DATA_DIR, "stream_a_normal")
STREAM_B_DIR = os.path.join(SFT_DATA_DIR, "stream_b_conflict")


def ensure_dirs():
    for d in [STREAM_A_DIR, STREAM_B_DIR]:
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# A-line: natural forensic pipeline
# ---------------------------------------------------------------------------

def collect_a_line_images(num_real: int, num_fake_per_class: int) -> List[tuple]:
    """Return list of (image_path, ground_truth) for A-line."""
    images = []

    real_files = sorted([
        f for f in os.listdir(REAL_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:num_real]
    for f in real_files:
        images.append((os.path.join(REAL_DIR, f), "Real"))

    for sub in GENIMAGE_SUBDIRS:
        sub_path = os.path.join(GENIMAGE_TEST_DIR, sub)
        if not os.path.isdir(sub_path):
            continue
        files = sorted([
            f for f in os.listdir(sub_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:num_fake_per_class]
        for f in files:
            images.append((os.path.join(sub_path, f), "Fake"))

    return images


def quality_check(session_data: dict) -> bool:
    """Return True if the session passes quality filters."""
    # Must have a verdict
    fv = session_data.get("final_verdict")
    if not fv or "verdict" not in fv:
        return False

    # Must have at least 1 evidence item (expert was called)
    if len(session_data.get("evidence_chain", [])) == 0:
        return False

    # Must have at least one <planning> and one <verdict> in conversations
    convs = session_data.get("conversations", [])
    has_planning = any("<planning>" in c.get("value", "") for c in convs if c.get("from") == "gpt")
    has_verdict = any("<verdict>" in c.get("value", "") for c in convs if c.get("from") == "gpt")
    if not has_planning or not has_verdict:
        return False

    return True


def run_a_line(fsm: ForensicStateMachine, images: List[tuple],
               logger: SessionLogger) -> dict:
    """Run natural forensic pipeline on all A-line images."""
    stats = {"total": len(images), "passed": 0, "failed": 0,
             "expert_calls": {"freq": 0, "noise": 0, "jpeg": 0},
             "verdicts": {"Real": 0, "Fake": 0, "Uncertain": 0},
             "total_steps": 0, "errors": 0}

    for i, (path, gt) in enumerate(images):
        fname = os.path.basename(path)
        try:
            result = fsm.run(path, ground_truth=gt)

            if quality_check(result):
                stats["passed"] += 1
            else:
                stats["failed"] += 1

            # Track expert calls
            for ev in result.get("evidence_chain", []):
                src = ev.get("source", "")
                if "freq" in src:
                    stats["expert_calls"]["freq"] += 1
                elif "noise" in src:
                    stats["expert_calls"]["noise"] += 1
                elif "jpeg" in src:
                    stats["expert_calls"]["jpeg"] += 1

            v = result.get("final_verdict", {}).get("verdict", "Uncertain")
            if v in stats["verdicts"]:
                stats["verdicts"][v] += 1

            stats["total_steps"] += result.get("total_steps", 0)

            status = "OK" if quality_check(result) else "FAIL"
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(images)}] {status}  "
                      f"pass={stats['passed']} fail={stats['failed']} "
                      f"err={stats['errors']}")

        except Exception as e:
            stats["errors"] += 1
            print(f"  [{i+1}/{len(images)}] ERROR: {fname} — {e}")

    stats["avg_steps"] = stats["total_steps"] / max(stats["passed"], 1)
    return stats


# ---------------------------------------------------------------------------
# B-line: constructed conflict / intervention scenarios
# ---------------------------------------------------------------------------

# Pre-built evidence tokens for B-line scenarios
FAKE_EVIDENCE_FREQ = json.dumps({
    "evidence_name": "high_frequency_grid_artifact",
    "region": "patch_coordinates_[204, 204, 819, 614]",
    "phenomenon": "The 2D-FFT power spectrum reveals concentrated periodic peaks "
                   "at (u=128, v=128) with amplitude 3.2σ above baseline.",
    "reasoning": "This spectral pattern is consistent with GAN/Diffusion "
                 "upsampling grid artifacts. Natural images do not exhibit "
                 "such structured high-frequency periodicity.",
    "strength": 0.92,
    "source": "frequency_expert",
    "support": "AI-generated",
    "interpretation_text": "Severe statistical anomaly matching artificial generative fingerprints.",
}, ensure_ascii=False)

REAL_EVIDENCE_NOISE = json.dumps({
    "evidence_name": "noise_consistency_normal",
    "region": "patch_coordinates_[306, 204, 716, 614]",
    "phenomenon": "Local noise variance measures 4.12, consistent with global "
                   "background variance 4.08 (±0.8%). No localised anomaly detected.",
    "reasoning": "The micro-noise pattern exhibits uniform variance across the region, "
                 "consistent with a single camera sensor capture without local editing.",
    "strength": 0.08,
    "source": "noise_expert",
    "support": "Real",
    "interpretation_text": "Statistical patterns align with normal hardware camera capture.",
}, ensure_ascii=False)

NEUTRAL_EVIDENCE = json.dumps({
    "evidence_name": "borderline_frequency_signal",
    "region": "patch_coordinates_[150, 150, 850, 700]",
    "phenomenon": "The frequency spectrum shows minor energy concentration at mid-high "
                   "frequencies, but below the statistical significance threshold (2.1σ).",
    "reasoning": "This could indicate either mild AI post-processing or natural image texture. "
                 "The signal is too weak for a definitive conclusion.",
    "strength": 0.48,
    "source": "frequency_expert",
    "support": "Uncertain",
    "interpretation_text": "Mild mathematical distortions noted; localized compression or blurring suspected.",
}, ensure_ascii=False)

SIMILAR_EVIDENCE_1 = json.dumps({
    "evidence_name": "borderline_signal_v1",
    "region": "patch_coordinates_[200, 200, 800, 600]",
    "phenomenon": "Minor frequency anomaly at 0.42σ above baseline.",
    "reasoning": "Weak signal, likely natural variation.",
    "strength": 0.45,
    "source": "frequency_expert",
    "support": "Uncertain",
    "interpretation_text": "Mild mathematical distortions noted.",
}, ensure_ascii=False)

SIMILAR_EVIDENCE_2 = json.dumps({
    "evidence_name": "borderline_signal_v2",
    "region": "patch_coordinates_[220, 180, 780, 620]",
    "phenomenon": "Minor frequency anomaly at 0.44σ above baseline.",
    "reasoning": "Weak signal, likely natural variation.",
    "strength": 0.46,
    "source": "frequency_expert",
    "support": "Uncertain",
    "interpretation_text": "Mild mathematical distortions noted.",
}, ensure_ascii=False)


def _extract_and_finalize(logger: SessionLogger, last_response: str):
    """Extract verdict from Qwen response and finalize the SFT session."""
    from utils.parser import Parser
    verdict = Parser.parse_verdict(last_response)
    if not verdict:
        # Qwen didn't output a verdict tag — construct a fallback
        verdict = {"verdict": "Uncertain", "confidence": 0.5,
                    "report": "模型未输出结构化 verdict，对话记录以 reasoning 为准。"}
    logger.finalize_sft(verdict, 5, "b_line_constructed")


def generate_b_line_scenarios(mllm, logger: SessionLogger, num_scenarios: int) -> dict:
    """
    Generate B-line data by constructing specific scenarios and feeding them
    to the Qwen model.  Three scenario types, evenly distributed.

    Returns stats dict.
    """
    stats = {"total": num_scenarios, "generated": 0, "errors": 0}
    per_type = num_scenarios // 3

    # Use a placeholder image path for B-line (image content doesn't matter —
    # the interesting part is how Qwen handles the evidence)
    placeholder_img = os.path.join(REAL_DIR, sorted(os.listdir(REAL_DIR))[0])

    scenario_types = [
        ("max_steps_exhaustion", per_type),
        ("evidence_conflict", per_type),
        ("info_gain_converged", per_type),
    ]

    seq = 0
    for scenario_name, count in scenario_types:
        for _ in range(count):
            seq += 1
            try:
                mllm.reset()
                logger.init_sft_session(
                    placeholder_img, "Fake", (512, 512), "b_line_constructed"
                )

                if scenario_name == "max_steps_exhaustion":
                    _scenario_max_steps(mllm, logger, placeholder_img)
                elif scenario_name == "evidence_conflict":
                    _scenario_conflict(mllm, logger, placeholder_img)
                elif scenario_name == "info_gain_converged":
                    _scenario_info_gain(mllm, logger, placeholder_img)

                # Finalize with a constructed verdict check
                logger.save_sft()
                stats["generated"] += 1

            except Exception as e:
                stats["errors"] += 1

            if seq % 30 == 0:
                print(f"  B-line [{seq}/{num_scenarios}] gen={stats['generated']} err={stats['errors']}")

    return stats


def _scenario_max_steps(mllm, logger, placeholder_img):
    """Simulate: Qwen keeps calling experts, budget runs out, forced to conclude."""
    # Turn 0: Qwen calls first expert
    resp0 = mllm.generate(placeholder_img, [])
    logger.add_conversation_turn("gpt", resp0)
    logger.add_evidence(json.loads(FAKE_EVIDENCE_FREQ))
    logger.add_conversation_turn("user", FAKE_EVIDENCE_FREQ)

    # Turn 1-3: keep injecting neutral evidence to simulate ongoing investigation
    for _ in range(3):
        logger.add_conversation_turn("user", NEUTRAL_EVIDENCE)
        logger.add_evidence(json.loads(NEUTRAL_EVIDENCE))

    # Budget exhausted message
    budget_msg = "[System: 取证资源（Budget）已耗尽，请立即基于已获取的全部证据撰写最终报告并输出 <verdict>。]"
    logger.add_system_message(budget_msg)

    resp_final = mllm.generate(placeholder_img, [
        {"from": "gpt", "value": resp0},
        {"from": "user", "value": FAKE_EVIDENCE_FREQ},
        {"from": "user", "value": budget_msg},
    ])
    logger.add_conversation_turn("gpt", resp_final)
    _extract_and_finalize(logger, resp_final)


def _scenario_conflict(mllm, logger, placeholder_img):
    """Simulate: freq says Fake (0.92), noise says Real (0.08) → conflict halt."""
    # Turn 0: Qwen calls freq
    resp0 = mllm.generate(placeholder_img, [])
    logger.add_conversation_turn("gpt", resp0)

    # Inject fake evidence (high strength)
    logger.add_evidence(json.loads(FAKE_EVIDENCE_FREQ))
    logger.add_conversation_turn("user", FAKE_EVIDENCE_FREQ)

    # Turn 1: Qwen reasons + might call another expert
    history = [
        {"from": "gpt", "value": resp0},
        {"from": "user", "value": FAKE_EVIDENCE_FREQ},
    ]
    resp1 = mllm.generate(placeholder_img, history)
    logger.add_conversation_turn("gpt", resp1)
    # Note: _extract_and_finalize called later after conflict resolution

    # Inject real evidence (low strength) — creates conflict
    logger.add_evidence(json.loads(REAL_EVIDENCE_NOISE))
    logger.add_conversation_turn("user", REAL_EVIDENCE_NOISE)

    # Conflict halt message
    conflict_msg = (
        "[System: 法证证据出现强冲突（疑罪从无 — Conflict Halting）。"
        "freq 专家强判 AI-generated (strength=0.92)，"
        "noise 专家强判 Real (strength=0.08)。"
        "请进行双向反思并输出 Uncertain 置信度校准结论。]"
    )
    logger.add_system_message(conflict_msg)

    history.append({"from": "gpt", "value": resp1})
    history.append({"from": "user", "value": REAL_EVIDENCE_NOISE})
    history.append({"from": "user", "value": conflict_msg})
    resp2 = mllm.generate(placeholder_img, history)
    logger.add_conversation_turn("gpt", resp2)
    _extract_and_finalize(logger, resp2)


def _scenario_info_gain(mllm, logger, placeholder_img):
    """Simulate: two nearly identical evidence tokens → info gain converged."""
    resp0 = mllm.generate(placeholder_img, [])
    logger.add_conversation_turn("gpt", resp0)

    logger.add_evidence(json.loads(SIMILAR_EVIDENCE_1))
    logger.add_conversation_turn("user", SIMILAR_EVIDENCE_1)

    # Second nearly-identical evidence
    logger.add_evidence(json.loads(SIMILAR_EVIDENCE_2))
    logger.add_conversation_turn("user", SIMILAR_EVIDENCE_2)

    # Info gain halt message
    gain_msg = (
        "[System: 信息增益收敛 — 连续两轮 Evidence Token 提供的新信息不足。"
        "请基于已有全部证据做出最终判定。]"
    )
    logger.add_system_message(gain_msg)

    history = [
        {"from": "gpt", "value": resp0},
        {"from": "user", "value": SIMILAR_EVIDENCE_1},
        {"from": "user", "value": SIMILAR_EVIDENCE_2},
        {"from": "user", "value": gain_msg},
    ]
    resp1 = mllm.generate(placeholder_img, history)
    logger.add_conversation_turn("gpt", resp1)
    _extract_and_finalize(logger, resp1)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def build_metadata(a_stats: dict, b_stats: dict, elapsed: float) -> dict:
    """Build aggregated metadata for the SFT dataset."""
    return {
        "generated_at": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "stream_a": a_stats,
        "stream_b": b_stats,
        "total_samples": a_stats.get("passed", 0) + b_stats.get("generated", 0),
        "total_failures": a_stats.get("failed", 0) + b_stats.get("errors", 0),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT data generation — Phase 2.3")
    parser.add_argument("--stream", type=str, default="both",
                        choices=["A", "B", "both"],
                        help="Which data stream to generate")
    parser.add_argument("--num-real", type=int, default=200,
                        help="Number of Real images for A-line")
    parser.add_argument("--num-fake", type=int, default=50,
                        help="Fake images per generator class for A-line")
    parser.add_argument("--num-scenarios", type=int, default=300,
                        help="Number of B-line constructed scenarios")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check setup without running GPU inference")
    args = parser.parse_args()

    ensure_dirs()

    print("=" * 60)
    print("SFT Data Generation — Phase 2.3")
    print(f"  Stream:     {args.stream}")
    print(f"  A-line:     {args.num_real} Real + {args.num_fake}×{len(GENIMAGE_SUBDIRS)} Fake")
    print(f"  B-line:     {args.num_scenarios} scenarios")
    print("=" * 60)

    if args.dry_run:
        print("\n[Dry-run] Checking setup...")
        imgs = collect_a_line_images(min(args.num_real, 2), min(args.num_fake, 1))
        print(f"  A-line images found: {len(imgs)}")
        print("  Setup OK. Remove --dry-run to start generation.")
        return

    t0 = time.time()

    # --- Load model once ---
    print("\n[1/4] Loading Qwen2.5-VL (once for all images)...")
    from mllm.qwen_client import QwenVLClient
    import torch
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. QwenVLClient requires GPU.")
        sys.exit(1)

    mllm = QwenVLClient()
    experts = {
        "frequency_expert": FrequencyExpert(),
        "noise_expert": NoiseExpert(),
        "jpeg_expert": JPEGExpert(),
    }
    fsm = ForensicStateMachine(mllm, experts)
    logger = SessionLogger()
    print("  Model loaded. Starting generation...")

    a_stats = {"passed": 0, "failed": 0, "errors": 0,
               "expert_calls": {}, "verdicts": {}, "avg_steps": 0}
    b_stats = {"total": 0, "generated": 0, "errors": 0}

    # --- A-line ---
    if args.stream in ("A", "both"):
        print(f"\n[2/4] A-line: natural forensic pipeline...")
        images = collect_a_line_images(args.num_real, args.num_fake)
        print(f"  Total images: {len(images)}")
        a_stats = run_a_line(fsm, images, logger)
        print(f"  A-line done. passed={a_stats['passed']} failed={a_stats['failed']} "
              f"errors={a_stats['errors']} avg_steps={a_stats.get('avg_steps', 0):.1f}")

    # --- B-line ---
    if args.stream in ("B", "both"):
        print(f"\n[3/4] B-line: constructed scenarios...")
        b_stats = generate_b_line_scenarios(mllm, logger, args.num_scenarios)
        print(f"  B-line done. generated={b_stats['generated']} errors={b_stats['errors']}")

    # --- Metadata ---
    print(f"\n[4/4] Building metadata...")
    elapsed = time.time() - t0
    meta = build_metadata(a_stats, b_stats, elapsed)
    meta_path = os.path.join(SFT_DATA_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata saved to: {meta_path}")

    # Summary
    total = meta["total_samples"]
    failures = meta["total_failures"]
    print("\n" + "=" * 60)
    print(f"GENERATION COMPLETE — {elapsed/60:.1f} min")
    print(f"  Total samples:  {total}")
    print(f"  Failures:       {failures}")
    print(f"  Output dir:     {SFT_DATA_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
