#!/usr/bin/env python3
"""
SFT training data construction — Phase 3.1a.

Classifies images into 4 types based on real Expert outputs + GT,
then synthesizes ideal Qwen forensic reasoning responses.

Types:
  1. 正确答案流 (~400): GT aligns with strongest Expert signal.
  2. 冲突反思流 (~200): Expert signals contradict each other.
  3. 格式示范流 (~100): Pure XML format reinforcement.
  4. 边界案例流 (~100): Expert strength in grey zone (0.25-0.75).

Usage:
  python scripts/build_sft_data.py --num-real 150 --num-fake-per-class 30
  python scripts/build_sft_data.py --dry-run
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    REAL_DIR, GENIMAGE_TEST_DIR, GENIMAGE_SUBDIRS, NORMALIZATION_SCALE,
)
from utils.image_utils import ImageUtils
from experts.frequency_v2 import FrequencyExpertV2
from experts.noise import NoiseExpert
from experts.jpeg import JPEGExpert

# Output directory
SFT_TRAIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "sft_data", "train",
)


# ---------------------------------------------------------------------------
# Data collection: run all 3 experts on each image
# ---------------------------------------------------------------------------

def collect_expert_outputs(image_paths: List[Tuple[str, str]]) -> List[dict]:
    """Run 3 experts on each image, return list of result dicts."""
    experts = {
        "freq": FrequencyExpertV2(),
        "noise": NoiseExpert(),
        "jpeg": JPEGExpert(),
    }
    results = []

    for i, (path, gt) in enumerate(image_paths):
        try:
            img = ImageUtils.load_image(path)
            expert_results = {}
            for name, expert in experts.items():
                r = expert.analyze(img)
                expert_results[name] = {
                    "strength": r.strength,
                    "support": r.support,
                    "phenomenon": r.phenomenon,
                    "reasoning": r.reasoning,
                    "interpretation_text": r.interpretation_text,
                    "evidence_name": r.evidence_name,
                    "source": r.source,
                }
            results.append({
                "image_path": path,
                "gt": gt,
                "experts": expert_results,
            })
        except Exception as e:
            pass  # skip corrupted images silently

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(image_paths)}] experts run...")

    return results


# ---------------------------------------------------------------------------
# Classification into 4 types
# ---------------------------------------------------------------------------

def classify_samples(samples: List[dict]) -> Dict[str, List[dict]]:
    """
    Classify each sample into one of 4 types:
      - correct: GT aligns with expert signal direction (relative ranking)
      - conflict: two experts strongly disagree (one > 0.6, another < 0.25)
      - borderline: key expert strength in [0.25, 0.6] grey zone
      - format: remaining samples (used for format training)
    """
    classified = {"correct": [], "conflict": [], "borderline": [], "format": []}

    for s in samples:
        exp = s["experts"]
        gt = s["gt"]
        strengths = {
            "freq": exp["freq"]["strength"],
            "noise": exp["noise"]["strength"],
            "jpeg": exp["jpeg"]["strength"],
        }

        best_expert = max(strengths, key=strengths.get)

        # Priority 1: Conflict — two experts STRONGLY disagree
        high_fake = [k for k, v in strengths.items() if v > 0.6]
        high_real = [k for k, v in strengths.items() if v < 0.25]
        if high_fake and high_real:
            classified["conflict"].append(s)
            continue

        # Priority 2: Correct — GT aligns with expert direction (use relative ranking)
        # For Fake GT: accept as "correct" to teach Fake verdict pattern
        # For Real GT: at least one expert clearly says Real
        if gt == "Fake":
            classified["correct"].append(s)
        elif gt == "Real" and (strengths[best_expert] < 0.5 or any(v < 0.3 for v in strengths.values())):
            classified["correct"].append(s)
        elif 0.25 <= strengths[best_expert] <= 0.6:
            classified["borderline"].append(s)
        else:
            classified["format"].append(s)

    # Cap each type
    return classified


# ---------------------------------------------------------------------------
# Template-based synthesis
# ---------------------------------------------------------------------------

def _gen_bbox_str(h: int, w: int, region: str = "center") -> str:
    """Generate a bbox string in [0, 1000] normalised coords."""
    if region == "center":
        ymin, xmin, ymax, xmax = 150, 150, 850, 850
    elif region == "face":
        ymin, xmin, ymax, xmax = 300, 200, 700, 800
    elif region == "top_left":
        ymin, xmin, ymax, xmax = 50, 50, 450, 450
    elif region == "bottom_right":
        ymin, xmin, ymax, xmax = 550, 550, 950, 950
    else:
        ymin, xmin, ymax, xmax = 200, 200, 800, 800
    return f"[{ymin}, {xmin}, {ymax}, {xmax}]"


def _evidence_json(exp: dict, bbox_str: str) -> str:
    """Build the Evidence Token JSON string that the state machine injects."""
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


def synthesize_correct(sample: dict) -> dict:
    """Synthesize type-1 (correct verdict) SFT sample."""
    exp = sample["experts"]
    gt = sample["gt"]
    bbox1 = _gen_bbox_str(512, 512, "center")
    bbox2 = _gen_bbox_str(512, 512, "face")

    # Choose primary expert: highest strength for Fake GT, lowest for Real GT
    strengths = {k: v["strength"] for k, v in exp.items()}
    if gt == "Fake":
        primary = max(strengths, key=strengths.get)
    else:
        primary = min(strengths, key=strengths.get)
    secondary = [k for k in strengths if k != primary][0]

    verdict = gt
    # Confidence: for Fake, proportional to best expert strength; for Real, inverse
    if gt == "Fake":
        confidence = round(min(0.95, strengths[primary] * 1.1 + 0.05), 4)
    else:
        confidence = round(min(0.95, (1.0 - strengths[primary]) * 1.1 + 0.05), 4)
    primary_ev = [exp[primary]["evidence_name"]]

    # Text for phenomenon summary
    prim_phen = exp[primary]["phenomenon"][:100]
    prim_str = strengths[primary]
    sec_phen = exp[secondary]["phenomenon"][:80]
    sec_str = strengths[secondary]

    conversations = [
        {"from": "user", "value": "<image>\n请分析这张图像的真实性，并使用法证工具箱开展多轮质证。"},
        {"from": "gpt", "value": (
            f"<planning>\n"
            f"Suspected Region: {bbox1}\n"
            f"Visual Anomalies: 图像区域存在需要法证验证的视觉特征。\n"
            f"Expert Target & Hypothesis: 拟调用 {primary} 专家进行微观物理痕迹分析。\n"
            f"</planning>\n"
            f"<call_{primary}>{bbox1}</call_{primary}>"
        )},
        {"from": "user", "value": _evidence_json(exp[primary], bbox1)},
        {"from": "gpt", "value": (
            f"<reasoning>\n"
            f"【物理-语义一致性校验】{primary} 专家反馈 {prim_phen}。"
            f"这与视觉层的观察在因果链上{'吻合' if prim_str > 0.4 else '存在关联'}。\n"
            f"已考虑图像后处理可能对物理指纹造成的衰减影响。\n"
            f"</reasoning>\n"
            f"<call_{secondary}>{bbox2}</call_{secondary}>"
        )},
        {"from": "user", "value": _evidence_json(exp[secondary], bbox2)},
        {"from": "gpt", "value": (
            f"<reasoning>\n"
            f"【综合判定】\n"
            f"{primary} 专家：{prim_phen}（strength={prim_str:.2f}，判定: {exp[primary]['support']}）\n"
            f"{secondary} 专家：{sec_phen}（strength={sec_str:.2f}，判定: {exp[secondary]['support']}）\n"
            f"综合多维度法证分析，{'两条证据链均指向' if exp[primary]['support'] == exp[secondary]['support'] else '主要证据综合判定'}"
            f"{'AI 生成' if verdict == 'Fake' else '真实相机'}来源。\n"
            f"</reasoning>\n"
            f"<verdict>\n"
            f"{json.dumps({'verdict': verdict, 'confidence': confidence, 'primary_evidence': primary_ev, 'report': f'经{primary}+{secondary}多轮法证分析，综合判定为{verdict}图像。'}, ensure_ascii=False)}\n"
            f"</verdict>"
        )},
    ]

    return {
        "id": f"sft_correct_{os.path.basename(sample['image_path'])}",
        "type": "correct",
        "image_path": sample["image_path"],
        "gt": gt,
        "conversations": conversations,
    }


def synthesize_conflict(sample: dict) -> dict:
    """Synthesize type-2 (conflict → Uncertain) SFT sample."""
    exp = sample["experts"]
    bbox1 = _gen_bbox_str(512, 512, "center")
    bbox2 = _gen_bbox_str(512, 512, "face")

    # Identify which experts are in conflict
    strengths = {k: v["strength"] for k, v in exp.items()}
    high_fake = sorted([k for k, v in strengths.items() if v > 0.7],
                       key=lambda k: -strengths[k])
    high_real = sorted([k for k, v in strengths.items() if v < 0.3],
                       key=lambda k: strengths[k])

    e1 = high_fake[0] if high_fake else "freq"
    e2 = high_real[0] if high_real else "noise"

    conversations = [
        {"from": "user", "value": "<image>\n请分析这张图像的真实性，并使用法证工具箱开展多轮质证。"},
        {"from": "gpt", "value": (
            f"<planning>\n"
            f"Suspected Region: {bbox1}\n"
            f"Visual Anomalies: 图像存在需要法证验证的不确定特征。\n"
            f"Expert Target & Hypothesis: 拟调用 {e1} 专家分析物理痕迹。\n"
            f"</planning>\n"
            f"<call_{e1}>{bbox1}</call_{e1}>"
        )},
        {"from": "user", "value": _evidence_json(exp[e1], bbox1)},
        {"from": "gpt", "value": (
            f"<reasoning>\n"
            f"【初步判断】{e1} 专家的发现需要进一步验证。"
            f"单一证据不足以定论——需要 {e2} 专家验证物理一致性。\n"
            f"</reasoning>\n"
            f"<call_{e2}>{bbox2}</call_{e2}>"
        )},
        {"from": "user", "value": _evidence_json(exp[e2], bbox2)},
        {"from": "gpt", "value": (
            f"<reasoning>\n"
            f"【证据冲突分析】\n"
            f"{e1} 专家{'强烈指出 AI 生成' if exp[e1]['strength'] > 0.7 else '信号不明确'}（strength={exp[e1]['strength']:.2f}），"
            f"但 {e2} 专家{'强烈指出真实相机' if exp[e2]['strength'] < 0.3 else '信号不明确'}（strength={exp[e2]['strength']:.2f}）。"
            f"两条证据在物理层面出现了根本性冲突。\n\n"
            f"【冲突溯源】可能原因：(1) 图像经过 AI 后处理——保留了部分原始特征但引入了 AI 痕迹；"
            f"(2) AI 生成图像叠加了后处理噪声——检测到生成痕迹但部分信号被掩盖。\n\n"
            f"【疑罪从无】当前证据不足以做出确定性判定，应给出 Uncertain 结论并附置信度校准。\n"
            f"</reasoning>\n"
            f"<verdict>\n"
            f"{json.dumps({'verdict': 'Uncertain', 'confidence': 0.45, 'primary_evidence': [], 'report': '法证证据在物理层面出现根本性冲突，疑罪从无。建议人工复核或使用更多维度进一步分析。'}, ensure_ascii=False)}\n"
            f"</verdict>"
        )},
    ]

    return {
        "id": f"sft_conflict_{os.path.basename(sample['image_path'])}",
        "type": "conflict",
        "image_path": sample["image_path"],
        "gt": sample["gt"],
        "conversations": conversations,
    }


def synthesize_borderline(sample: dict) -> dict:
    """Synthesize type-3 (borderline → cautious verdict) SFT sample."""
    exp = sample["experts"]
    gt = sample["gt"]
    bbox1 = _gen_bbox_str(512, 512, "center")

    # Find the borderline expert
    borderline_exp = "noise"
    for k, v in exp.items():
        if 0.25 <= v["strength"] <= 0.75:
            borderline_exp = k
            break

    confidence = round(0.65 + np.random.uniform(0, 0.12), 4)

    conversations = [
        {"from": "user", "value": "<image>\n请分析这张图像的真实性，并使用法证工具箱开展多轮质证。"},
        {"from": "gpt", "value": (
            f"<planning>\n"
            f"Suspected Region: {bbox1}\n"
            f"Visual Anomalies: 图像部分区域存在轻微异常。\n"
            f"Expert Target & Hypothesis: 调用 {borderline_exp} 专家确认微观信号。\n"
            f"</planning>\n"
            f"<call_{borderline_exp}>{bbox1}</call_{borderline_exp}>"
        )},
        {"from": "user", "value": _evidence_json(exp[borderline_exp], bbox1)},
        {"from": "gpt", "value": (
            f"<reasoning>\n"
            f"【边界分析】{borderline_exp} 专家返回 strength={exp[borderline_exp]['strength']:.2f}"
            f"（Uncertain 区间）。这一结果处于真实特征与异常特征之间的灰色地带。\n"
            f"自然场景中的正常波动也可能导致类似偏差。在没有其他专家给出强信号的情况下，"
            f"倾向判定为{gt}，但置信度相应降低以反映边界不确定性。\n"
            f"</reasoning>\n"
            f"<verdict>\n"
            f"{json.dumps({'verdict': gt, 'confidence': confidence, 'primary_evidence': [], 'report': f'{borderline_exp}专家分析结果处于灰色地带。综合视觉特征倾向判定为{gt}，置信度较低反映边界不确定性。'}, ensure_ascii=False)}\n"
            f"</verdict>"
        )},
    ]

    return {
        "id": f"sft_borderline_{os.path.basename(sample['image_path'])}",
        "type": "borderline",
        "image_path": sample["image_path"],
        "gt": gt,
        "conversations": conversations,
    }


def synthesize_format(sample: dict) -> dict:
    """Synthesize type-4 (format-only) SFT sample."""
    exp = sample["experts"]
    bbox = _gen_bbox_str(512, 512, "center")
    expert = np.random.choice(["freq", "noise", "jpeg"])

    conversations = [
        {"from": "user", "value": "<image>\n请分析这张图像的真实性，并使用法证工具箱开展多轮质证。"},
        {"from": "gpt", "value": (
            f"<planning>\n"
            f"Suspected Region: {bbox}\n"
            f"Visual Anomalies: 需法证分析的视觉特征。\n"
            f"Expert Target & Hypothesis: 调用 {expert} 专家。\n"
            f"</planning>\n"
            f"<call_{expert}>{bbox}</call_{expert}>"
        )},
        {"from": "user", "value": _evidence_json(exp[expert], bbox)},
        {"from": "gpt", "value": (
            f"<reasoning>\n"
            f"【物理-语义一致性校验】{expert} 专家的分析结果已纳入综合评估。\n"
            f"</reasoning>\n"
            f"<verdict>\n"
            f"{json.dumps({'verdict': exp[expert]['support'], 'confidence': round(exp[expert]['strength'], 4), 'primary_evidence': [exp[expert]['evidence_name']], 'report': f'基于{expert}专家法证分析的综合判定。'}, ensure_ascii=False)}\n"
            f"</verdict>"
        )},
    ]

    return {
        "id": f"sft_format_{os.path.basename(sample['image_path'])}",
        "type": "format",
        "image_path": sample["image_path"],
        "gt": sample["gt"],
        "conversations": conversations,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build SFT training data — Phase 3.1a")
    parser.add_argument("--num-real", type=int, default=150,
                        help="Number of Real images to process")
    parser.add_argument("--num-fake-per-class", type=int, default=30,
                        help="Fake images per GenImage subdirectory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check setup without running")
    parser.add_argument("--output-dir", type=str, default=SFT_TRAIN_DIR)
    args = parser.parse_args()

    print("=" * 60)
    print("SFT Data Construction — Phase 3.1a")
    print(f"  Real: {args.num_real}")
    print(f"  Fake/class: {args.num_fake_per_class}")
    print(f"  Total Fake: ~{args.num_fake_per_class * len(GENIMAGE_SUBDIRS)}")
    print("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Collect image paths
    print("\n[1/3] Collecting images...")
    image_paths = []
    real_files = sorted([
        f for f in os.listdir(REAL_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:args.num_real]
    for f in real_files:
        image_paths.append((os.path.join(REAL_DIR, f), "Real"))

    for sub in GENIMAGE_SUBDIRS:
        sub_path = os.path.join(GENIMAGE_TEST_DIR, sub)
        if not os.path.isdir(sub_path):
            continue
        files = sorted([
            f for f in os.listdir(sub_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])[:args.num_fake_per_class]
        for f in files:
            image_paths.append((os.path.join(sub_path, f), "Fake"))

    print(f"  Total images: {len(image_paths)}")

    if args.dry_run:
        print("  Dry-run complete. Remove --dry-run to build data.")
        return

    # 2. Run all 3 experts
    print("\n[2/3] Running experts on all images...")
    t0 = time.time()
    samples = collect_expert_outputs(image_paths)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — {len(samples)} valid samples")

    # 3. Classify into 4 types
    print("\n[3/3] Classifying and synthesizing...")
    classified = classify_samples(samples)

    for typ, data in classified.items():
        print(f"  {typ}: {len(data)} samples")

    # Synthesize and save
    synthesizers = {
        "correct": synthesize_correct,
        "conflict": synthesize_conflict,
        "borderline": synthesize_borderline,
        "format": synthesize_format,
    }

    total = 0
    targets = {"correct": 400, "conflict": 200, "borderline": 100, "format": 100}

    for typ, data in classified.items():
        synthesizer = synthesizers[typ]
        target = targets.get(typ, len(data))
        selected = data[:target]

        output_file = os.path.join(args.output_dir, f"sft_{typ}.json")
        sft_samples = []
        for s in selected:
            try:
                sft_record = synthesizer(s)
                sft_samples.append(sft_record)
            except Exception as e:
                pass

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(sft_samples, f, ensure_ascii=False, indent=2)
        print(f"  → {output_file}: {len(sft_samples)} records")
        total += len(sft_samples)

    print(f"\n  Total SFT samples: {total}")
    print(f"  Output: {args.output_dir}/")


if __name__ == "__main__":
    main()
