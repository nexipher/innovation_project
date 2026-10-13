#!/usr/bin/env python3
"""
G7 §4.17 A2: the calibration audit the handoff asks for, from cached scores.

Five questions, answered from what is already on disk — no model runs, and no
refitting anywhere near C or the test partition:

  1. Where does each expert's mapping come from (cells, sample counts, label
     ratios, bin edges) and which single cell decides its weight?
  2. How much of the old table's fitting set belongs to B and C?
  3. Why does a q85 JPEG token carry the 0.972 reliability, and what does that
     number actually measure?
  4. Empirical bins vs smoothed bins vs logistic regression, compared on Brier
     and reliability with source-level uncertainty — not on AUROC alone.
  5. Does the reference posterior depend on the order experts were called in?

The comparison in (4) fits every candidate on **A only** and evaluates on B,
so the three are judged on the same footing.  C and test stay untouched.
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

from config import PROJECT_ROOT
from utils.batch import load_trajectory_records

TABLE_PATH = os.path.join(PROJECT_ROOT, "calibration", "reliability_table.json")
G2_REPORT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g2_expert_report.json")
RAW_VALUES_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "raw_values.json")
MANIFEST_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "manifest.json")
SPLIT_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "g7_split.json")
PROBE_TEMPLATE = os.path.join(PROJECT_ROOT, "calibration", "g7_external_probe_dino_{group}.json")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_calibration_audit.json")
TRACES_DIR = os.path.join(PROJECT_ROOT, "sft_data", "trajectories_g7_b1")

MATCHED_CELLS = ("real_png", "fake_png", "real_jpeg_q95", "fake_jpeg_q95",
                 "real_jpeg_q85", "fake_jpeg_q85", "real_jpeg_q70", "fake_jpeg_q70")
G2_TO_SOURCE = {"noise": "noise_expert", "jpeg": "jpeg_expert",
                "frequency_v2": "frequency_expert_v2", "ela": "ela_expert"}
BOOTSTRAP_ROUNDS = 400


def _load(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# 1-3: provenance, overlap, and the single cell behind a weight
# ---------------------------------------------------------------------------

def provenance() -> dict:
    table = _load(TABLE_PATH)
    g2 = _load(G2_REPORT_PATH)
    manifest = {s["sample_id"]: s for s in _load(MANIFEST_PATH)["samples"]}
    groups = _load(SPLIT_PATH)["groups"]
    source_group = {sid: name for name, ids in groups.items() for sid in ids}
    raw = _load(RAW_VALUES_PATH)["experts"]

    output = {}
    for g2_name, source in G2_TO_SOURCE.items():
        if source not in table["experts"]:
            continue
        entry = table["experts"][source]
        cells = g2["experts"].get(g2_name, {}).get("cell_metrics", {})
        per_cell = {}
        for cell, metrics in sorted(cells.items()):
            auroc = metrics.get("auroc_fake_vs_real")
            corrected = None if auroc is None else max(auroc, 1.0 - auroc)
            per_cell[cell] = {"auroc_fake_vs_real": auroc,
                              "separation_polarity_corrected": corrected,
                              "n_real": metrics.get("n_real"), "n_fake": metrics.get("n_fake")}

        fit_ids = [sid for sid in raw.get(g2_name, {}).get("raw", {})
                   if manifest.get(sid) and manifest[sid]["cell"] in MATCHED_CELLS]
        by_group = Counter(source_group.get(manifest[sid]["source_id"], "?")
                           for sid in fit_ids)
        labels = Counter(manifest[sid]["label"] for sid in fit_ids)

        output[source] = {
            "g2_name": g2_name,
            "fits_on_cells": list(MATCHED_CELLS),
            "n_fit_samples": len(fit_ids),
            "fit_labels": dict(labels),
            "fit_by_g7_group": dict(by_group),
            "bins": [{"lo": b["lo"], "hi": b["hi"], "n": b["n"],
                      "p_fake": round(b["p_fake"], 4)} for b in entry["bins"]],
            # Which single number the runtime weight comes from: `png` only,
            # because evaluate_experts_g2.py sets
            # separation_polarity_corrected = max(png, 1 - png).
            "weight": {"source_cell": "png",
                       "value": round(float(entry["separation_polarity_corrected"]), 4),
                       "youden_margin": round(max(0.0, 2.0 * entry["separation_polarity_corrected"] - 1.0), 4)},
            "per_cell_separation": per_cell,
            "applicability": entry["applicability"],
        }

    # PROBE enters from its own report, fitted on A by construction.
    probe_entry = table["experts"].get("probe_dino_expert")
    if probe_entry:
        provenance_block = dict(probe_entry.get("provenance") or {})
        output["probe_dino_expert"] = {
            "g2_name": None,
            "fits_on_cells": provenance_block.get("treatments"),
            "n_fit_samples": probe_entry["n_samples"],
            "fit_labels": None,
            "fit_by_g7_group": {"A": probe_entry["n_samples"]},
            "bins": [{"lo": b["lo"], "hi": b["hi"], "n": b["n"],
                      "p_fake": round(b["p_fake"], 4)} for b in probe_entry["bins"]],
            "weight": {"source_cell": "pooled matched cells",
                       "value": round(float(probe_entry["separation_polarity_corrected"]), 4),
                       "youden_margin": round(max(0.0, 2.0 * probe_entry["separation_polarity_corrected"] - 1.0), 4)},
            "per_cell_separation": {k: {"separation_polarity_corrected": v}
                                    for k, v in (provenance_block.get("per_treatment_separation") or {}).items()},
            "applicability": probe_entry["applicability"],
        }
    return output


def sample_series(g2_name: str, group: str) -> Tuple[List[float], List[int], List[str]]:
    """(values, labels, source_ids) for one expert on one G7 group."""
    manifest = {s["sample_id"]: s for s in _load(MANIFEST_PATH)["samples"]}
    members = set(_load(SPLIT_PATH)["groups"][group])
    raw = _load(RAW_VALUES_PATH)["experts"][g2_name]["raw"]
    values, labels, sources = [], [], []
    for sid, value in raw.items():
        meta = manifest.get(sid)
        if meta is None or meta["source_id"] not in members:
            continue
        if meta["cell"] not in MATCHED_CELLS or value is None:
            continue
        values.append(float(value))
        labels.append(1 if meta["label"] == "Fake" else 0)
        sources.append(meta["source_id"])
    return values, labels, sources


def probe_series(group: str) -> Tuple[List[float], List[int], List[str]]:
    path = PROBE_TEMPLATE.format(group=group)
    if not os.path.exists(path):
        return [], [], []
    rows = _load(path)["rows"]
    values = [float(r["score"]) for r in rows]
    labels = [1 if r["label"] == "Fake" else 0 for r in rows]
    sources = [r["source_id"] for r in rows]
    return values, labels, sources


# ---------------------------------------------------------------------------
# 4: three mappings, fitted on A, judged on B
# ---------------------------------------------------------------------------

def empirical_bins(values: List[float], labels: List[int], bins: int = 5) -> List[dict]:
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    output = []
    for index in range(len(edges) - 1):
        low, high = float(edges[index]), float(edges[index + 1])
        mask = (np.asarray(values) >= low) & (np.asarray(values) < high)
        if index == len(edges) - 2:
            mask = np.asarray(values) >= low
        if mask.sum() == 0:
            continue
        output.append({"lo": None if index == 0 else low,
                       "hi": None if index == len(edges) - 2 else high,
                       "n": int(mask.sum()),
                       "p_fake": float(np.mean(np.asarray(labels)[mask]))})
    return output


def bin_probability(bins: List[dict], value: float) -> float:
    for bucket in bins:
        low, high = bucket["lo"], bucket["hi"]
        if low is not None and value < low:
            continue
        if high is not None and value >= high:
            continue
        return float(bucket["p_fake"])
    return 0.5


def smoothed_bins(bins: List[dict], base_rate: float, alpha: float = 1.0) -> List[dict]:
    """Shrink every bin toward the base rate: 0/1 empirical probabilities lie."""
    output = []
    for bucket in bins:
        n, p = bucket["n"], bucket["p_fake"]
        output.append({**bucket, "p_fake": (p * n + base_rate * alpha) / (n + alpha)})
    return output


def fit_logistic(values: List[float], labels: List[int]):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(values, dtype=float).reshape(-1, 1)
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(X), labels)
    return scaler, model


def evaluate_mapping(values: List[float], labels: List[int], sources: List[str],
                     predict) -> dict:
    probabilities = [float(predict(v)) for v in values]
    brier = float(np.mean([(p - y) ** 2 for p, y in zip(probabilities, labels)]))

    # reliability: mean |predicted − empirical| over 5 equal-count bins
    order = np.argsort(probabilities)
    chunks = np.array_split(order, 5)
    gaps = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        mean_pred = float(np.mean([probabilities[i] for i in chunk]))
        mean_true = float(np.mean([labels[i] for i in chunk]))
        gaps.append(abs(mean_pred - mean_true))

    # source-level uncertainty on the Brier score
    by_source: Dict[str, List[int]] = defaultdict(list)
    for index, source in enumerate(sources):
        by_source[source].append(index)
    keys = sorted(by_source)
    rng = random.Random(7)
    scores = []
    for _ in range(BOOTSTRAP_ROUNDS):
        picked = [rng.choice(keys) for _ in keys]
        idx = [i for source in picked for i in by_source[source]]
        scores.append(np.mean([(probabilities[i] - labels[i]) ** 2 for i in idx]))
    scores.sort()
    return {
        "brier": round(brier, 4),
        "brier_ci": [round(scores[int(0.025 * len(scores))], 4),
                     round(scores[int(0.975 * len(scores)) - 1], 4)],
        "reliability_gap": round(float(np.mean(gaps)), 4) if gaps else None,
        "n": len(values),
        "sources": len(keys),
    }


def compare_mappings() -> dict:
    output = {}
    for g2_name, source in list(G2_TO_SOURCE.items()) + [("probe_dino", "probe_dino_expert")]:
        if g2_name == "ela":
            continue
        a_values, a_labels, _ = (probe_series("A") if g2_name == "probe_dino"
                                 else sample_series(g2_name, "A"))
        b_values, b_labels, b_sources = (probe_series("B") if g2_name == "probe_dino"
                                         else sample_series(g2_name, "B"))
        if not a_values or not b_values:
            continue
        base_rate = float(np.mean(a_labels))
        bins = empirical_bins(a_values, a_labels)
        smooth = smoothed_bins(bins, base_rate, alpha=2.0)
        scaler, model = fit_logistic(a_values, a_labels)

        entry = {
            "fit_on": "A", "evaluated_on": "B",
            "fit_samples": len(a_values), "eval_samples": len(b_values),
            "empirical_bins": evaluate_mapping(
                b_values, b_labels, b_sources, lambda v: bin_probability(bins, v)),
            "smoothed_bins": evaluate_mapping(
                b_values, b_labels, b_sources, lambda v: bin_probability(smooth, v)),
            "logistic": evaluate_mapping(
                b_values, b_labels, b_sources,
                lambda v: float(model.predict_proba(scaler.transform([[v]]))[0][1])),
            "empirical_values": {"bins": len(bins),
                                 "zeros_or_ones": sum(1 for bucket in bins
                                                      if bucket["p_fake"] in (0.0, 1.0))},
        }
        for name in ("empirical_bins", "smoothed_bins", "logistic"):
            entry[name].pop("sources", None)
        output[source] = entry
    return output


# ---------------------------------------------------------------------------
# 5: does the reference posterior depend on call order?
# ---------------------------------------------------------------------------

def order_dependence(traces_dir: str = TRACES_DIR) -> dict:
    from state_machine.halting_v2 import HaltingPolicyV2

    records = load_trajectory_records(traces_dir)
    deltas = []
    for record in records.values():
        chain = [{"source": t.get("source"),
                  "reliability": 2.0 * (1.0 - 0.0) if False else None,
                  "calibrated_likelihood": t.get("calibrated_likelihood")}
                 for t in (record.get("evidence") or [])]
        if len(chain) < 2:
            continue
        # The runtime weight is the table's separation per source.
        table = _load(TABLE_PATH)["experts"]
        for token in chain:
            entry = table.get(token["source"]) or {}
            token["reliability"] = entry.get("separation_polarity_corrected")
        forward = HaltingPolicyV2.posterior(chain)
        backward = HaltingPolicyV2.posterior(list(reversed(chain)))
        deltas.append(abs(forward - backward))
    if not deltas:
        return {"sessions_with_two_tokens": 0}
    deltas.sort()
    return {
        "sessions_with_two_tokens": len(deltas),
        "max_abs_delta": round(deltas[-1], 4),
        "median_abs_delta": round(deltas[len(deltas) // 2], 4),
        "sessions_over_0.01": sum(1 for d in deltas if d > 0.01),
        "note": ("相关性折减施加在“后调用者”身上，因此后验依赖调用顺序；"
                 "这里量化的是同一组证据在记录顺序与逆序下的后验差"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--traces", default=TRACES_DIR)
    args = parser.parse_args()

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provenance": provenance(),
        "mapping_comparison_fit_A_eval_B": compare_mappings(),
        "order_dependence": order_dependence(args.traces),
        "constraints": ("只用缓存分数；未用 C 或最终测试集拟合或调参；"
                        "C 组与 test 分区未参与本审计的任何计算"),
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1, allow_nan=False)

    print("== 拟合来源与权重来源 ==")
    for source, block in report["provenance"].items():
        groups = block["fit_by_g7_group"]
        print(f"  {source:22s} n={block['n_fit_samples']:4d} 分组={groups} "
              f"权重来自 {block['weight']['source_cell']}={block['weight']['value']}")
    print("\n== 三种映射（A 拟合 → B 评估）Brier / 可靠性差 ==")
    header = f"  {'专家':22s} {'经验分箱':>18s} {'平滑分箱':>18s} {'逻辑回归':>18s}"
    print(header)
    for source, block in report["mapping_comparison_fit_A_eval_B"].items():
        cells = []
        for name in ("empirical_bins", "smoothed_bins", "logistic"):
            e = block[name]
            cells.append(f"{e['brier']:.3f} [{e['brier_ci'][0]:.3f},{e['brier_ci'][1]:.3f}]")
        print(f"  {source:22s} {cells[0]:>18s} {cells[1]:>18s} {cells[2]:>18s}")
    print("\n== 顺序依赖 ==")
    print("  " + json.dumps(report["order_dependence"], ensure_ascii=False))
    print(f"\nReport: {os.path.relpath(args.output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
