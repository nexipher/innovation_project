#!/usr/bin/env python3
"""
Finalize SFT training dataset — Phase 3.1a (complete).

Steps:
  1. Filter A-line samples where verdict == GT (~140 samples).
  2. Generate borderline samples from bench-set images.
  3. Generate format samples from any source.
  4. Output consolidated dataset to sft_data/train/final/.

Usage:
  python scripts/finalize_sft_data.py
  python scripts/finalize_sft_data.py --dry-run
"""

import json
import os
import sys
import glob
import random
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    REAL_DIR, GENIMAGE_TEST_DIR, GENIMAGE_SUBDIRS, NORMALIZATION_SCALE,
)
from utils.image_utils import ImageUtils
from experts.frequency_v2 import FrequencyExpertV2
from experts.noise import NoiseExpert
from experts.jpeg import JPEGExpert

SFT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sft_data", "train", "final",
)
TRACES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "traces", "sft_sessions",
)


# =========================================================================
# Step 1: Filter A-line correct samples
# =========================================================================

def filter_a_line_correct() -> List[dict]:
    """
    Scan A-line SFT files from Phase 2.3. Keep only those where
    final_verdict.verdict == ground_truth AND quality checks pass.
    """
    print("\n[Step 1] Filtering A-line correct samples...")

    a_files = sorted(glob.glob(os.path.join(TRACES_DIR, "forensic_sft_session_20260721_1[1-2]*.json")))
    print(f"  A-line files found: {len(a_files)}")

    correct = []
    for f in a_files:
        try:
            d = json.load(open(f))
        except:
            continue

        # Quality checks
        v = d.get("final_verdict") or {}
        if not v or "verdict" not in v:
            continue
        gt = d.get("ground_truth", "")
        if not gt or gt not in ("Real", "Fake"):
            continue
        if v["verdict"] != gt:
            continue
        if len(d.get("evidence_chain", [])) == 0:
            continue

        # Tag the sample
        d["type"] = "correct_filtered"
        d["source"] = "phase2_a_line"
        correct.append(d)

    print(f"  Verdict == GT: {len(correct)}")
    return correct


# =========================================================================
# Step 2: Generate borderline samples
# =========================================================================

def gen_bbox_str(region="center"):
    if region == "center":
        return "[150, 150, 850, 850]"
    elif region == "face":
        return "[300, 200, 700, 800]"
    else:
        return "[200, 200, 800, 800]"


def evidence_json(exp: dict, bbox_str: str) -> str:
    return json.dumps({
        "evidence_name": exp["evidence_name"],
        "region": f"patch_coordinates_{bbox_str}",
        "phenomenon": exp["phenomenon"],
        "reasoning": exp["reasoning"],
        "strength": round(exp["strength"], 4),
        "source": exp["source"],
        "support": exp["support"],
        "interpretation_text": exp["interpretation_text"],
    }, ensure_ascii=False)


def generate_borderline(num_samples: int = 100) -> List[dict]:
    """
    Run experts on benchmark set, pick images where any expert strength
    is in [0.25, 0.6], synthesize cautious-reasoning samples.
    """
    print(f"\n[Step 2] Generating borderline samples (target={num_samples})...")

    # Collect images
    images = []
    real_files = sorted([f for f in os.listdir(REAL_DIR) if f.lower().endswith((".jpg",".jpeg",".png"))])[:80]
    for f in real_files:
        images.append((os.path.join(REAL_DIR, f), "Real"))
    for sub in GENIMAGE_SUBDIRS:
        sp = os.path.join(GENIMAGE_TEST_DIR, sub)
        if not os.path.isdir(sp):
            continue
        files = sorted([f for f in os.listdir(sp) if f.lower().endswith((".jpg",".jpeg",".png"))])[:20]
        for f in files:
            images.append((os.path.join(sp, f), "Fake"))

    experts = {
        "freq": FrequencyExpertV2(),
        "noise": NoiseExpert(),
        "jpeg": JPEGExpert(),
    }

    borderline = []
    for path, gt in images:
        # Run experts
        try:
            img = ImageUtils.load_image(path)
            exp = {}
            for name, expert in experts.items():
                r = expert.analyze(img)
                exp[name] = {
                    "strength": r.strength, "support": r.support,
                    "phenomenon": r.phenomenon, "reasoning": r.reasoning,
                    "interpretation_text": r.interpretation_text,
                    "evidence_name": r.evidence_name, "source": r.source,
                }
        except:
            continue

        # Check if any expert is in grey zone
        strengths = {k: v["strength"] for k, v in exp.items()}
        grey_experts = [k for k, v in strengths.items() if 0.25 <= v <= 0.6]
        if not grey_experts:
            continue

        borderline_exp = grey_experts[0]
        bbox = gen_bbox_str("center")

        verdict = gt  # borderline still leans toward GT
        confidence = round(0.62 + random.uniform(0, 0.12), 4)

        sample = {
            "id": f"sft_borderline_{os.path.basename(path).split('.')[0]}",
            "type": "borderline",
            "image_path": path,
            "gt": gt,
            "source": "synthesized",
            "conversations": [
                {"from": "user", "value": "<image>\n请分析这张图像的真实性，并使用法证工具箱开展多轮质证。"},
                {"from": "gpt", "value": (
                    f"<planning>\n"
                    f"Suspected Region: {bbox}\n"
                    f"Visual Anomalies: 图像存在需要法证验证的细微特征。\n"
                    f"Expert Target & Hypothesis: 调用 {borderline_exp} 专家分析微观物理信号。\n"
                    f"</planning>\n"
                    f"<call_{borderline_exp}>{bbox}</call_{borderline_exp}>"
                )},
                {"from": "user", "value": evidence_json(exp[borderline_exp], bbox)},
                {"from": "gpt", "value": (
                    f"<reasoning>\n"
                    f"【边界分析】{borderline_exp} 专家返回 strength={strengths[borderline_exp]:.2f}，"
                    f"处于真实特征与异常特征之间的灰色地带（0.25-0.60）。\n"
                    f"这一结果本身不能{'确证' if gt == 'Fake' else '排除'}AI 生成——"
                    f"自然场景中的光照变化、纹理复杂度差异以及图像格式差异"
                    f"（JPEG vs PNG）也可能导致类似偏差。\n"
                    f"在无其他专家给出强信号的情况下，"
                    f"倾向判定为{gt}，但置信度相应降低以反映边界不确定性。\n"
                    f"</reasoning>\n"
                    f"<verdict>\n"
                    f"{json.dumps({'verdict': gt, 'confidence': confidence, 'primary_evidence': [], 'report': f'{borderline_exp}专家分析结果处于灰色地带（strength={strengths[borderline_exp]:.2f}）。综合视觉特征倾向判定为{gt}，置信度较低反映边界不确定性。'}, ensure_ascii=False)}\n"
                    f"</verdict>"
                )},
            ],
        }
        borderline.append(sample)
        if len(borderline) >= num_samples:
            break

    print(f"  Generated: {len(borderline)}")
    return borderline


# =========================================================================
# Step 3: Generate format samples
# =========================================================================

def generate_format(num_samples: int = 100) -> List[dict]:
    """
    Extract format-only samples from any source. These teach XML structure
    regardless of content correctness. Use A-line/B-line samples that
    have complete conversations but may have wrong verdicts.
    """
    print(f"\n[Step 3] Generating format samples (target={num_samples})...")

    # Source 1: A-line files with complete format (even if verdict wrong)
    a_files = sorted(glob.glob(os.path.join(TRACES_DIR, "forensic_sft_session_20260721_1[1-2]*.json")))

    format_samples = []
    for f in a_files:
        try:
            d = json.load(open(f))
        except:
            continue
        convs = d.get("conversations", [])
        if len(convs) < 4:
            continue  # need planning + evidence + reasoning + verdict
        # Check format completeness
        gpt_text = " ".join(c.get("value", "") for c in convs if c.get("from") == "gpt")
        has_p = "<planning>" in gpt_text
        has_c = "<call_" in gpt_text
        has_r = "<reasoning>" in gpt_text
        has_v = "<verdict>" in gpt_text
        if not (has_p and has_c and has_r and has_v):
            continue

        d["type"] = "format"
        d["source"] = "phase2_filtered"
        format_samples.append(d)

        if len(format_samples) >= num_samples:
            break

    print(f"  Generated: {len(format_samples)}")
    return format_samples


# =========================================================================
# Main
# =========================================================================

def main():
    os.makedirs(SFT_DATA_DIR, exist_ok=True)

    random.seed(42)

    print("=" * 60)
    print("Finalize SFT Training Data")
    print("=" * 60)

    # Step 1
    correct = filter_a_line_correct()

    # Step 2
    borderline = generate_borderline(100)

    # Step 3
    format_samples = generate_format(100)

    # Load existing conflict data
    conflict_file = os.path.join(os.path.dirname(SFT_DATA_DIR), "sft_conflict.json")
    conflict = []
    if os.path.exists(conflict_file):
        conflict = json.load(open(conflict_file))
    print(f"\n[Existing] conflict: {len(conflict)} (loaded from file)")

    # ====== Consolidate ======
    print("\n" + "=" * 60)
    print("Final Dataset Summary")
    print("=" * 60)

    datasets = {
        "correct": correct,
        "conflict": conflict[:200],   # cap
        "borderline": borderline[:100],
        "format": format_samples[:100],
    }

    total = 0
    for name, data in datasets.items():
        output_file = os.path.join(SFT_DATA_DIR, f"sft_{name}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {name}: {len(data)} → {output_file}")
        total += len(data)

    print(f"\n  TOTAL: {total} samples → {SFT_DATA_DIR}/")

    # Metadata
    meta = {
        "generated_at": "2026-07-21",
        "total": total,
        "breakdown": {k: len(v) for k, v in datasets.items()},
        "sources": {
            "correct": "A-line filtered (verdict==GT)",
            "conflict": "synthesized from bench-set",
            "borderline": "synthesized from grey-zone bench-set",
            "format": "A-line format-complete (content may be wrong)",
        },
    }
    with open(os.path.join(SFT_DATA_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Metadata: {os.path.join(SFT_DATA_DIR, 'metadata.json')}")


if __name__ == "__main__":
    main()
