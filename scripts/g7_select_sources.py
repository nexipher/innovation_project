#!/usr/bin/env python3
"""
G7 entry 4, step 2a: choose the sources for the first training batch.

This runs the *selection* and nothing else.  It scores a candidate pool with
PROBE on the card (the score depends on the encoding, so the variants are built
first), maps each score to the calibrated band the runtime would use, and picks
12 sources.  No trajectories are generated here.

Candidate pool: train partition only, asserted disjoint from B/C/D, from the
acceptance batch and from the test partition — the same discipline the batches
themselves are held to.

The selection rule, written before the scores arrive:

  * 5 Real + 7 Fake, the Fake spread across as many generators as possible;
  * a source is *useful* when its five variants cover more than one band — a
    confident reading and an ambiguous one in the same photograph is exactly
    the situation that produces disagreement and honest abstention, and those
    are outcomes the selection can only make room for, never order up;
  * prefer sources with both a strong and a mid-band variant, then generator
    diversity, then lowest source id (deterministic tie-break);
  * Real sources get the same treatment: a confident-Real variant and a
    mid-band one, so the Real side produces sessions that actually call tools
    rather than sailing to "Real" without evidence.

Usage:
  python scripts/g7_select_sources.py --dry-run          # pool + variants only
  python scripts/g7_select_sources.py                    # needs GPU (PROBE)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import GENIMAGE_SUBDIRS, PROJECT_ROOT
from utils.batch import batch_paths

SPLIT_PATH = os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
MANIFEST_D_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "manifest_d.json")
ACCEPTANCE_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "acceptance_batch.json")
SPLIT_ABC_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "g7_split.json")

DEFAULT_BATCH = "g7_b1"
DEFAULT_SEED = 20261011
REAL_CANDIDATES = 40
FAKE_PER_GENERATOR = 6

# Band edges on the calibrated P(Fake) — the same reading the runtime table
# gives a token.  `mid` is the near-chance band the weak-evidence check uses.
BANDS = (("strong_fake", 0.9, 1.01), ("lean_fake", 0.6, 0.9),
         ("mid", 0.4, 0.6), ("lean_real", 0.1, 0.4), ("strong_real", -0.01, 0.1))

SELECTION = {"Real": 6, "Fake": 8}

# Rule amendment, 2026-10-11, *after* the first selection was scored.  Recorded
# here rather than in a commit message because it is part of what this batch is.
#
#   seen        the 12-source rule (5 Real + 7 Fake) left Wukong out: all six of
#               its candidates read strong_fake in every one of their five
#               formats, so they lose the preference for band spread and the
#               Fake quota was full.  The candidate pool also contained Real
#               sources whose five formats all landed in the near-chance band.
#   changed     quota 5+7 -> 6+8, so every generator is represented and a
#               pure-mid-bin Real enters the batch.
#   why allowed this is candidate *selection* for a training set, not an
#               independent test; the user's words: hard cases may be covered
#               on purpose.  It is a revision made with the scores in hand and
#               is described as one — not as a rule that anticipated them.
#   what it is  NOT a claim that the mid-bin source should be abstained on.
#               The mid band says PROBE is unsure; whether the session ends in
#               a verdict, a disagreement or an abstention is an outcome to
#               observe, and the corroborating experts may well carry it.
#
# The 12-source selection is kept beside the new one
# (`selection_v1_12sources.json`) so the revision stays visible.
AMENDMENT = {"date": "2026-10-11", "from": {"Real": 5, "Fake": 7},
             "to": dict(SELECTION), "seen": "Wukong 全部候选强信号、池中存在全中位箱 Real",
             "reason": "覆盖 8 个生成器 + 纳入一个全中位箱 Real；属于看过分数后的规则修订"}


def band_of(probability: float) -> str:
    for name, low, high in BANDS:
        if low <= probability < high:
            return name
    return "unknown"


def used_paths() -> set:
    """Every image that belongs to a group that has already been decided."""
    used = set()
    for path in (MANIFEST_D_PATH, ACCEPTANCE_PATH):
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            used |= {s["source_path"] for s in json.load(handle)["samples"]}
    with open(SPLIT_ABC_PATH, encoding="utf-8") as handle:
        groups = json.load(handle)["groups"]
    with open(os.path.join(PROJECT_ROOT, "calibration", "set", "manifest.json"),
              encoding="utf-8") as handle:
        manifest = json.load(handle)["samples"]
    members = set(groups["A"]) | set(groups["B"]) | set(groups["C"])
    used |= {s["source_path"] for s in manifest if s["source_id"] in members}
    return used


def select_candidates(seed: int = DEFAULT_SEED) -> List[dict]:
    """Train-partition sources, excluding everything already spoken for."""
    with open(SPLIT_PATH, encoding="utf-8") as handle:
        split = json.load(handle)
    blocked = used_paths()
    rng = random.Random(seed)

    train = {key: entry for key, entry in split["sources"].items()
             if entry["split"] == "train" and entry["path"] not in blocked}
    real = sorted(key for key, entry in train.items() if entry["label"] == "Real")
    rng.shuffle(real)
    candidates = [{"source_id": key, **train[key], "origin": "train"}
                  for key in real[:REAL_CANDIDATES]]

    for generator in GENIMAGE_SUBDIRS:
        pool = sorted(key for key, entry in train.items()
                      if entry["generator"] == generator)
        rng.shuffle(pool)
        candidates.extend({"source_id": key, **train[key], "origin": "train"}
                          for key in pool[:FAKE_PER_GENERATOR])
    return candidates


def check_candidates(candidates: List[dict]) -> List[str]:
    problems = []
    with open(SPLIT_PATH, encoding="utf-8") as handle:
        split = json.load(handle)
    test_paths = {e["path"] for e in split["sources"].values() if e["split"] == "test"}
    blocked = used_paths()
    for source in candidates:
        if source["split"] != "train":
            problems.append(f"{source['source_id']} 不在训练分区")
        if source["path"] in test_paths or source["path"] in blocked:
            problems.append(f"{source['source_id']} 与已用批次/最终测试重叠")
    if len({s["source_id"] for s in candidates}) != len(candidates):
        problems.append("候选池内有重复源")
    return problems


def build_variants(candidates: List[dict], batch: str) -> List[dict]:
    """
    The four encoded cells per source (the native format is the source file).

    Returned as a list: four variants share one source_id, and keying them by
    it (the first version of this function) silently kept only the last.
    """
    from scripts.build_variants_g4 import build_variants as build

    paths = batch_paths(batch)
    with open(SPLIT_PATH, encoding="utf-8") as handle:
        split = json.load(handle)
    manifest = build([s["source_id"] for s in candidates], split,
                     out_dir=paths["variants_dir"])
    return [{**entry, "_path": key} for key, entry in manifest.items()]


def score_variants(candidates: List[dict], variants: List[dict],
                   device: str = "cuda") -> List[dict]:
    """PROBE over every candidate image: source (native) plus its four cells."""
    import cv2
    from experts.probe_dino import ProbeDinoExpert
    from utils.reliability import ReliabilityTable

    expert = ProbeDinoExpert(device=device)
    reason = expert.unavailable_reason()
    if reason:
        raise SystemExit(f"PROBE 不可用：{reason}")
    table = ReliabilityTable.load()

    rows: List[dict] = []
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for entry in variants:
        by_source[entry["source_id"]].append(entry)

    for index, source in enumerate(candidates, 1):
        images = [("native", source["path"])] + \
                 sorted(((e["treatment"], e["_path"]) for e in
                         by_source.get(source["source_id"], [])),
                        key=lambda pair: pair[0])
        for treatment, path in images:
            image = cv2.imread(os.path.join(PROJECT_ROOT, path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            score = float(expert.analyze(image).raw_metric)
            lookup = table.lookup("probe_dino_expert", score) or {}
            probability = (lookup.get("calibrated_likelihood") or {}).get("Fake")
            rows.append({
                "source_id": source["source_id"], "label": source["label"],
                "generator": source["generator"], "treatment": treatment,
                "path": path, "raw_score": round(score, 6),
                "p_fake": None if probability is None else round(float(probability), 4),
                "band": band_of(float(probability)) if probability is not None else "unknown",
            })
        if index % 20 == 0:
            print(f"    打分 [{index}/{len(candidates)}] 源")
    return rows


def summarise_per_source(rows: List[dict]) -> List[dict]:
    by_source: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    summary = []
    for source_id, entries in by_source.items():
        bands = [entry["band"] for entry in entries]
        counts = Counter(bands)
        summary.append({
            "source_id": source_id,
            "label": entries[0]["label"],
            "generator": entries[0]["generator"],
            "bands": dict(counts),
            "band_set": sorted(set(bands)),
            "n_strong": counts["strong_fake"] + counts["strong_real"],
            "n_mid": counts["mid"],
            "spread": len(set(bands)) > 1,
        })
    return summary


def signal_label_association(rows: List[dict]) -> dict:
    """
    Does the selection make signal strength track the label?

    A batch where every Fake reads strong and every Real reads weak would teach
    "the tool is loud ⇒ Fake" — the shortcut G2-d spent its budget removing.
    Hard cases are allowed on purpose here (the user said so), but the
    *distribution* that results must not be mistaken for a natural one, so the
    chosen set is reported next to the pool it came from.
    """
    bands_by_label: Dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        bands_by_label[row["label"]][row["band"]] += 1

    output = {}
    for label in ("Real", "Fake"):
        counts = bands_by_label[label]
        total = sum(counts.values())
        strong = counts["strong_fake"] + counts["strong_real"]
        output[label] = {
            "n": total,
            "strong_share": round(strong / total, 4) if total else None,
            "mid_share": round(counts["mid"] / total, 4) if total else None,
            "bands": {name: counts[name] for name, _, _ in BANDS},
        }
    fake = output["Fake"]["strong_share"]
    real = output["Real"]["strong_share"]
    output["strong_share_gap"] = None if fake is None or real is None else round(fake - real, 4)
    return output


def choose(summary: List[dict]) -> List[dict]:
    """
    The pre-registered rule, in the order it was written down.

    Preference: a source whose own variants disagree about the band (both
    readings in one photograph) > one that only ever reads the same way; then
    Fake across generators, so the batch does not learn one generator's
    signature.  Ties break on the source id, so two runs choose the same set.
    """
    chosen: List[dict] = []
    used: Counter = Counter()

    for label in ("Fake", "Real"):
        quota = SELECTION[label]
        pool = sorted((entry for entry in summary if entry["label"] == label),
                      key=lambda entry: (
                          not (entry["n_strong"] and entry["n_mid"]),
                          not entry["spread"],
                          -entry["n_strong"],
                          entry["source_id"]))
        taken = 0
        if label == "Fake":
            # first pass: one Fake per generator; second pass fills the rest
            for entry in pool:
                if taken >= quota:
                    break
                if used[entry["generator"]] >= 1:
                    continue
                chosen.append(entry)
                used[entry["generator"]] += 1
                taken += 1
            for entry in pool:
                if taken >= quota:
                    break
                if entry in chosen:
                    continue
                chosen.append(entry)
                used[entry["generator"]] += 1
                taken += 1
        else:
            for entry in pool:
                if taken >= quota:
                    break
                chosen.append(entry)
                taken += 1
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", default=DEFAULT_BATCH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true",
                        help="只建候选池与变体（CPU），不打分、不选源")
    args = parser.parse_args()

    paths = batch_paths(args.batch)
    candidates = select_candidates(args.seed)
    problems = check_candidates(candidates)
    if problems:
        raise SystemExit("候选池不合法：" + "；".join(problems))
    print(f"候选池：{len(candidates)} 源（Real "
          f"{sum(1 for s in candidates if s['label'] == 'Real')} / Fake "
          f"{sum(1 for s in candidates if s['label'] == 'Fake')}）"
          f"｜生成器 {len({s['generator'] for s in candidates})} 个")

    variants = build_variants(candidates, args.batch)
    print(f"变体：{len(variants)} 个 → {os.path.relpath(paths['variants_dir'], PROJECT_ROOT)}/"
          f"（每源 4 个编码格 + 源图本身 = 5 个格式）")

    if args.dry_run:
        print("（dry-run：未打分、未选源）")
        return

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA 不可用；PROBE 打分需要 GPU（agent.md §3.2）。")
    rows = score_variants(candidates, variants)
    summary = summarise_per_source(rows)
    chosen = choose(summary)
    chosen_ids = {entry["source_id"] for entry in chosen}

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "batch": args.batch,
        "seed": args.seed,
        "pool": {"sources": len(candidates),
                 "labels": dict(Counter(s["label"] for s in candidates))},
        "selection_rule": {
            "quota": SELECTION,
            "prefer": "同一源的五个格式跨分带（既有强信号又有中位箱）→ 生成器分散 → 源 id",
            "bands": {name: [low, high] for name, low, high in BANDS},
        },
        "amendment": AMENDMENT,
        "association": {
            "chosen": signal_label_association(
                [row for row in rows if row["source_id"] in chosen_ids]),
            "pool": signal_label_association(rows),
            "note": ("选定集有意覆盖难例，其分布不等于自然分布；这里只把差异量化，"
                     "不作'应当弃权'之类的预设"),
        },
        "superseded": "selection_v1_12sources.json（5 Real + 7 Fake 的原方案）",
        "chosen": sorted(chosen_ids),
        "chosen_detail": sorted(chosen, key=lambda e: (e["label"], e["source_id"])),
        "per_source": sorted(summary, key=lambda e: e["source_id"]),
        "variants": rows,
    }
    # The batch's generation input: only the chosen sources' variants.  The
    # candidate manifest is kept beside it — the selection is evidence, and it
    # has to stay checkable.
    chosen_variants = {v["_path"]: {k: val for k, val in v.items() if k != "_path"}
                       for v in variants if v["source_id"] in chosen_ids}
    with open(os.path.join(paths["variants_dir"], "manifest_candidates.json"),
              "w", encoding="utf-8") as handle:
        json.dump({"generated_at": report["generated_at"], "pool": len(candidates),
                   "counts": {"sources": len(candidates), "variants": len(variants)},
                   "variants": {v["_path"]: {k: val for k, val in v.items()
                                             if k != "_path"} for v in variants}},
                  handle, ensure_ascii=False, indent=1)
    with open(paths["variants"], "w", encoding="utf-8") as handle:
        json.dump({"generated_at": report["generated_at"],
                   "split_hash": json.load(open(SPLIT_PATH, encoding="utf-8"))["hash"],
                   "partition": "train", "batch": args.batch,
                   "counts": {"sources": len(chosen_ids), "variants": len(chosen_variants)},
                   "variants": chosen_variants}, handle, ensure_ascii=False, indent=1)

    output = os.path.join(paths["samples"], "selection.json")
    os.makedirs(paths["samples"], exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1)

    print(f"\n=== 分带表（按源；每源 5 个格式）===")
    print(f"{'源':34s} {'类':4s} {'生成器':12s} " + " ".join(
        f"{name:>11s}" for name, _, _ in BANDS) + "  跨带")
    for entry in sorted(summary, key=lambda e: (e["label"], e["source_id"])):
        mark = "★" if entry["source_id"] in chosen_ids else " "
        counts = entry["bands"]
        print(f"{mark}{entry['source_id'][:32]:33s} {entry['label']:4s} "
              f"{entry['generator']:12s} "
              + " ".join(f"{counts.get(name, 0):11d}" for name, _, _ in BANDS)
              + f"  {'是' if entry['spread'] else '—'}")

    print(f"\n=== 选定 {len(chosen)} 源 ===")
    for entry in sorted(chosen, key=lambda e: (e["label"], e["source_id"])):
        print(f"  {entry['label']:4s} {entry['generator']:12s} {entry['source_id']:40s} "
              f"分带 {entry['bands']}")
    chosen_assoc = report["association"]["chosen"]
    pool_assoc = report["association"]["pool"]
    print(f"\n=== 信号-标签关联（强信号占比）===")
    for name, stats in (("候选池", pool_assoc), ("选定 14 源", chosen_assoc)):
        print(f"  {name:10s} Real {stats['Real']['strong_share']:.2f}"
              f"（中位 {stats['Real']['mid_share']:.2f}）｜"
              f"Fake {stats['Fake']['strong_share']:.2f}"
              f"（中位 {stats['Fake']['mid_share']:.2f}）｜"
              f"差 {stats['strong_share_gap']:+.2f}")
    print("  注：选定集有意覆盖难例，分布不等同自然分布，仅作对照记录")
    print(f"\n清单：{os.path.relpath(output, PROJECT_ROOT)}")
    print("（本轮只选源；未生成任何轨迹）")


if __name__ == "__main__":
    main()
