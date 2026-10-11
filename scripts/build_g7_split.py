#!/usr/bin/env python3
"""
G7-1: the source-level split the external experts are fitted, selected and
accepted on.

The review's rule: fitting, choosing and accepting must not share data, and
the split is by **source image** — the same photograph in another format is
the same source and cannot straddle two groups.  The project's test partition
(`split_v2`) is untouched and is what the final paired comparison uses; this
script only divides the 98 calibration sources that `split_v2` holds out.

    A  fitting     probability calibration, reliability table, applicability
    B  development tool set, thresholds, call costs, fusion hyper-parameters
    C  validation  read-only re-check before anything is accepted

Usage:
  python scripts/build_g7_split.py                 # write calibration/set/g7_split.json
  python scripts/build_g7_split.py --dry-run       # report only
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "manifest.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "g7_split.json")
DEFAULT_SEED = "g7-2026-10-11"
DEFAULT_RATIOS = {"A": 0.6, "B": 0.2, "C": 0.2}
GROUPS = ("A", "B", "C")


def load_sources(manifest_path: str = MANIFEST_PATH) -> Dict[str, dict]:
    """One entry per source image, with the treatments it contributes."""
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    sources: Dict[str, dict] = {}
    for sample in manifest["samples"]:
        entry = sources.setdefault(sample["source_id"], {
            "label": sample["label"], "generator": sample["generator"],
            "treatments": set(), "cells": set(), "samples": [],
            "resolution": sample.get("resolution_bucket"),
        })
        entry["treatments"].add(sample["treatment"])
        entry["cells"].add(sample["cell"])
        entry["samples"].append(sample["sample_id"])
    for entry in sources.values():
        entry["treatments"] = sorted(entry["treatments"])
        entry["cells"] = sorted(entry["cells"])
        entry["samples"] = sorted(entry["samples"])
    return sources


def _rank(source_id: str, seed: str) -> str:
    """A stable ordering inside a stratum: no RNG state to reproduce."""
    return hashlib.sha256(f"{seed}|{source_id}".encode("utf-8")).hexdigest()


def assign(sources: Dict[str, dict], seed: str = DEFAULT_SEED,
           ratios: Optional[Dict[str, float]] = None) -> Dict[str, List[str]]:
    """
    Stratified by (label, generator), assigned within each stratum by a stable
    hash order, so the split is reproducible from the seed alone and every
    generator appears in every group.
    """
    ratios = ratios or DEFAULT_RATIOS
    strata: Dict[tuple, List[str]] = {}
    for source_id, entry in sources.items():
        strata.setdefault((entry["label"], entry["generator"]), []).append(source_id)

    groups: Dict[str, List[str]] = {name: [] for name in GROUPS}
    for stratum in sorted(strata):
        members = sorted(strata[stratum], key=lambda s: _rank(s, seed))
        counts = {name: int(len(members) * ratios[name]) for name in GROUPS}
        # The remainder goes to the fitting group: it can afford the extra
        # source, the two smaller groups cannot.
        counts["A"] += len(members) - sum(counts.values())
        cursor = 0
        for name in GROUPS:
            groups[name].extend(members[cursor:cursor + counts[name]])
            cursor += counts[name]
    for name in groups:
        groups[name].sort()
    return groups


def validate(groups: Dict[str, List[str]], sources: Dict[str, dict]) -> List[str]:
    problems: List[str] = []
    seen = [s for name in GROUPS for s in groups[name]]
    if len(seen) != len(set(seen)):
        problems.append("重叠：同源出现在多个组")
    if set(seen) != set(sources):
        problems.append("不完整：有源未被分配")
    for name in GROUPS:
        if not groups[name]:
            problems.append(f"{name} 组为空")
        labels = {sources[s]["label"] for s in groups[name]}
        if labels != {"Real", "Fake"}:
            problems.append(f"{name} 组缺少类别：{sorted(labels)}")
        generators = {sources[s]["generator"] for s in groups[name] if sources[s]["label"] == "Fake"}
        if len(generators) < 8:
            problems.append(f"{name} 组缺少生成器：{sorted(generators)}")
    for name, target in (("A", 0.6), ("B", 0.2), ("C", 0.2)):
        actual = len(groups[name]) / max(len(sources), 1)
        if abs(actual - target) > 0.08:
            problems.append(f"{name} 组比例偏离目标：{actual:.2f} vs {target}")
    return problems


def fingerprint(groups: Dict[str, List[str]]) -> str:
    canonical = json.dumps({name: sorted(groups[name]) for name in GROUPS},
                           ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build(manifest_path: str = MANIFEST_PATH, output_path: str = OUTPUT_PATH,
          seed: str = DEFAULT_SEED, dry_run: bool = False) -> dict:
    sources = load_sources(manifest_path)
    groups = assign(sources, seed)
    problems = validate(groups, sources)
    if problems:
        raise SystemExit("划分不合法：" + "；".join(problems))

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "ratios": DEFAULT_RATIOS,
        "hash": fingerprint(groups),
        "counts": {name: len(groups[name]) for name in GROUPS},
        "samples": {name: sum(len(sources[s]["samples"]) for s in groups[name])
                    for name in GROUPS},
        "labels": {name: _label_counts(groups[name], sources) for name in GROUPS},
        "generators": {name: sorted({sources[s]["generator"] for s in groups[name]})
                       for name in GROUPS},
        "groups": groups,
        "sources": {source_id: {**entry, "group": next(
            name for name in GROUPS if source_id in groups[name])}
            for source_id, entry in sorted(sources.items())},
        "note": ("最终测试使用 split_v2 的 test 分区，与此划分互斥；"
                 "拟合/选择/验收分别只读 A/B/C 组"),
    }
    if not dry_run:
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=1)
    return report


def _label_counts(members: List[str], sources: Dict[str, dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for source_id in members:
        label = sources[source_id]["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = build(args.manifest, args.output, args.seed, args.dry_run)
    print(f"hash {report['hash']}  源 {sum(report['counts'].values())}  "
          f"样本 {sum(report['samples'].values())}")
    for name in GROUPS:
        print(f"  {name} 组：源 {report['counts'][name]:3d}  "
              f"样本 {report['samples'][name]:4d}  {report['labels'][name]}  "
              f"生成器 {len(report['generators'][name])} 个")
    print("（末行：训练/最终评测用 split_v2 的 test 分区，与此互斥）")
    if not args.dry_run:
        print(f"Report: {os.path.relpath(args.output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
