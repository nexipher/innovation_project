#!/usr/bin/env python3
"""
"Does the second expert actually add?" — paired bootstrap on one partition.

The comparison script reports overlap and a naive average; this puts a
confidence interval on the *difference*, because on a 284-source cell a +0.02
AUROC can be an artefact or a fact and the point estimate alone cannot say
which.

**Resampling is by source, not by image.**  Each of the 284 val sources appears
as four variants (png/q70/q85/q95); drawing images independently would treat one
source's four encodings as four independent samples and shrink the interval by
roughly sqrt(4) — the CI would look tighter than the data supports.  The unit
the split is defined on is the source, so the unit of resampling is the source:
a draw takes a source and **all** of its variants that the cell contains.
(Inside a single-format cell each source contributes exactly one image, so a
cluster draw and an image draw coincide there: those cells were always sound.)

Equal weights, no fitting: an unfitted rule can be frozen as-is, whereas a
fitted one would have to be fitted on train/val and only then tested.

Usage:
  python scripts/g8_gain_bootstrap.py --base probe --add npr-a univfd noise \
      --split val --out sft_data/g8_eval/gain_bootstrap_val.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import PROJECT_ROOT
from utils.score_semantics import entry as semantics_entry


def latest_run(expert: str, split: str) -> str:
    base = os.path.join(PROJECT_ROOT, "sft_data", "g8_eval")
    matches = []
    for name in sorted(os.listdir(base)):
        report = os.path.join(base, name, "report.json")
        if not os.path.exists(report) or not name.endswith(f"_{split}"):
            continue
        with open(report, encoding="utf-8") as handle:
            data = json.load(handle)
        if data["expert"] == expert and not data.get("mock"):
            matches.append(os.path.join(base, name))
    if not matches:
        raise SystemExit(f"没有 {expert} 在 {split} 上的评测运行")
    return matches[-1]


def load_scores(run_dir: str) -> Dict[str, dict]:
    rows = {}
    with open(os.path.join(run_dir, "scores.jsonl"), encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("raw_score") is not None:
                rows[row["variant_id"]] = row
    return rows


def decision_scores(expert: str, run_dir: str, ids: List[str]) -> np.ndarray:
    rows = load_scores(run_dir)
    with open(os.path.join(run_dir, "report.json"), encoding="utf-8") as handle:
        source_name = json.load(handle)["source_name"]
    direction = semantics_entry(source_name)["raw_score"]["direction"]
    raw = np.array([rows[i]["raw_score"] for i in ids], dtype=float)
    return raw if direction == "higher_is_fake" else 1.0 - raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="probe")
    parser.add_argument("--add", nargs="+", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261015)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from sklearn.metrics import roc_auc_score

    base_dir = latest_run(args.base, args.split)
    runs = {name: latest_run(name, args.split) for name in args.add}
    base_rows = load_scores(base_dir)
    ids = sorted(set(base_rows) & set.intersection(
        *[set(load_scores(path)) for path in runs.values()]))
    if not ids:
        raise SystemExit("运行之间没有共同图像")

    labels = np.array([1 if base_rows[i]["label"] == "Fake" else 0 for i in ids])
    treatments = np.array([base_rows[i]["treatment"] for i in ids])
    sources = np.array([base_rows[i]["source_id"] for i in ids])
    base = decision_scores(args.base, base_dir, ids)
    cells = ["ALL"] + sorted(set(treatments.tolist()))
    rng = np.random.default_rng(args.seed)

    report = {"split": args.split, "base": args.base, "images": len(ids),
              "sources": int(len(set(sources.tolist()))),
              "resampling_unit": "source（连同该源在本格内的全部变体）",
              "base_auroc": float(roc_auc_score(labels, base)), "cells": {}}
    for name in args.add:
        added = decision_scores(name, runs[name], ids)
        report["cells"][name] = {}
        for cell in cells:
            mask = np.ones(len(ids), bool) if cell == "ALL" else (treatments == cell)
            y, a, b = labels[mask], base[mask], added[mask]
            cell_sources = sources[mask]
            unique = sorted(set(cell_sources.tolist()))
            index_of = {source: np.where(cell_sources == source)[0] for source in unique}
            deltas = []
            for _ in range(args.resamples):
                drawn = rng.choice(len(unique), len(unique), replace=True)
                index = np.concatenate([index_of[unique[i]] for i in drawn])
                if len(set(y[index].tolist())) < 2:
                    continue
                deltas.append(roc_auc_score(y[index], (a[index] + b[index]) / 2)
                              - roc_auc_score(y[index], a[index]))
            values = np.array(deltas)
            # 每次抽到的源数也是数据的一部分：单源在格的 n 不同要看得见
            per_source = len(y) / max(len(unique), 1)
            report["cells"][name][cell] = {
                "n": int(mask.sum()),
                "sources": len(unique),
                "images_per_source": round(per_source, 3),
                "base_auroc": float(roc_auc_score(y, a)),
                "average_auroc": float(roc_auc_score(y, (a + b) / 2)),
                "delta_mean": float(values.mean()),
                "ci95": [float(np.percentile(values, 2.5)),
                         float(np.percentile(values, 97.5))],
                "p_delta_le_zero": float((values <= 0).mean()),
            }
            row = report["cells"][name][cell]
            print(f"  {args.base}+{name:8} {cell:9} n={row['n']:4} "
                  f"（{row['sources']} 源）Δ={row['delta_mean']:+.4f} "
                  f"CI[{row['ci95'][0]:+.4f},{row['ci95'][1]:+.4f}] "
                  f"P(Δ≤0)={row['p_delta_le_zero']:.3f}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"报告：{os.path.relpath(args.out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
