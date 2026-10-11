#!/usr/bin/env python3
"""
G7 entry 2: the three baselines, and the question they exist to answer.

    现有 noise/jpeg/freq 融合    旧工具箱本身有多强？
    PROBE 单独                   新专家本身有多强？
    现有三专家 + PROBE 融合      旧专家还能否提供额外价值？

Group discipline (plan.md §4.16 G7-1), enforced rather than promised:

    A  fit        the logistic-regression fusion, its probability calibration
                  and the candidate thresholds
    B  select     which condition to keep, and the operating threshold
    C  validate   scored once the scheme is frozen — the run refuses until
                  `calibration/g7_frozen_scheme.json` exists and matches the
                  split hash

The fusion is deliberately a **logistic regression** on standardised scores:
four features against ~460 samples does not support anything grander, and a
model that cannot be written down in a freeze file cannot be validated on C.
The fit also decides each expert's *sign* — no polarity is pre-imposed, which
is how the existing experts' inversions get discovered rather than assumed.

Cached scores mean most of this is CPU: existing experts come from
`calibration/set/raw_values.json`, PROBE from its A/B/C reports.

Usage:
  python scripts/g7_baselines.py --fit-select          # A + B, writes the freeze file
  python scripts/g7_baselines.py --validate-c          # C, only after freezing
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "manifest.json")
RAW_VALUES_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "raw_values.json")
SPLIT_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "g7_split.json")
FREEZE_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_frozen_scheme.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "calibration")

EXISTING_EXPERTS = ("noise", "jpeg", "frequency_v2")   # the shipped toolkit
PROBE_SOURCE = "probe_dino_expert"
CONDITIONS = {
    "existing": list(EXISTING_EXPERTS),
    "probe": [PROBE_SOURCE],
    "existing+probe": list(EXISTING_EXPERTS) + [PROBE_SOURCE],
}
TREATMENTS = ("native", "png", "jpeg_q95", "jpeg_q85", "jpeg_q70")
BOOTSTRAP_ROUNDS = 2000
DEFAULT_THRESHOLD = 0.5

# Step 3's decision rule, written down before the numbers arrive so the
# numbers cannot choose it.
FUSION_WINS_MARGIN = 0.02      # ΔAUROC on B needed to call the fusion better
FUSION_LOSES_MARGIN = -0.02


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_group(group: str) -> Tuple[List[str], dict]:
    with open(SPLIT_PATH, encoding="utf-8") as handle:
        split = json.load(handle)
    if group not in ("A", "B", "C"):
        raise SystemExit(f"unknown group {group!r}")
    return [s for s in split["groups"][group]], split


def load_feature_table(group: str) -> Tuple[List[str], Dict[str, Dict[str, float]], Dict[str, dict]]:
    """
    {sample_id: {feature: score}} for one group, from the caches.

    PROBE's sealed report is accepted here as a *table only*: this function
    never aggregates, so loading C cannot leak a metric.
    """
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(RAW_VALUES_PATH, encoding="utf-8") as handle:
        cached = json.load(handle)

    members, _ = load_group(group)
    member_set = set(members)
    samples = {s["sample_id"]: s for s in manifest["samples"]
               if s["source_id"] in member_set}

    table: Dict[str, Dict[str, float]] = {sid: {} for sid in samples}
    for expert in EXISTING_EXPERTS:
        raw = (cached.get("experts", {}).get(expert) or {}).get("raw", {})
        for sid in samples:
            value = raw.get(sid)
            if value is not None:
                table[sid][expert] = float(value)

    probe = _probe_rows(group)
    for sid, score in probe.items():
        if sid in table:
            table[sid][PROBE_SOURCE] = score
    return sorted(samples), table, samples


def _probe_rows(group: str) -> Dict[str, float]:
    """PROBE scores for a group, from whichever report holds them."""
    candidates = [os.path.join(OUTPUT_DIR, f"g7_external_probe_dino_{group}.json"),
                  os.path.join(OUTPUT_DIR, f"g7_external_probe_dino_{group}_sealed.json")]
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        if report.get("expert") != "probe_dino":
            continue
        return {row["sample_id"]: float(row["score"]) for row in report["rows"]}
    raise SystemExit(f"找不到 {group} 组的 PROBE 分数（先跑 evaluate_external_expert.py）")


def matrix(sample_ids: List[str], table: Dict[str, Dict[str, float]],
           features: List[str], medians: Optional[Dict[str, float]] = None):
    """(X, medians) with A-group medians filling a missing expert's value."""
    if medians is None:
        medians = {}
        for feature in features:
            values = [table[sid][feature] for sid in sample_ids if feature in table[sid]]
            medians[feature] = float(np.median(values)) if values else 0.0
    X = np.array([[table[sid].get(feature, medians[feature]) for feature in features]
                  for sid in sample_ids], dtype=float)
    return X, medians


# ---------------------------------------------------------------------------
# Metrics (CPU, no model)
# ---------------------------------------------------------------------------

def auroc(probabilities, labels) -> Optional[float]:
    positive = [p for p, y in zip(probabilities, labels) if y == 1]
    negative = [p for p, y in zip(probabilities, labels) if y == 0]
    if len(positive) < 3 or len(negative) < 3:
        return None
    scores = np.concatenate([positive, negative])
    ranks = rankdata(scores)
    n = len(positive)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * len(negative)))


def _acc_f1(probabilities, labels, threshold: float) -> Tuple[float, float]:
    predicted = [1 if p >= threshold else 0 for p in probabilities]
    correct = sum(1 for p, y in zip(predicted, labels) if p == y)
    tp = sum(1 for p, y in zip(predicted, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(predicted, labels) if p == 1 and y == 0)
    fn = sum(1 for p, y in zip(predicted, labels) if p == 0 and y == 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return correct / len(labels), f1


def brier(probabilities, labels) -> float:
    return float(np.mean([(p - y) ** 2 for p, y in zip(probabilities, labels)]))


def best_threshold(probabilities, labels) -> Tuple[float, float]:
    """The threshold maximising F1 on the selection group (B picks this)."""
    grid = sorted({round(p, 3) for p in probabilities} | {DEFAULT_THRESHOLD})
    best = (DEFAULT_THRESHOLD, -1.0)
    for threshold in grid:
        _, f1 = _acc_f1(probabilities, labels, threshold)
        if f1 > best[1]:
            best = (threshold, f1)
    return best[0], round(best[1], 4)


def summarise(probabilities, labels, sample_ids, samples, threshold) -> dict:
    per_treatment: Dict[str, dict] = {}
    for treatment in TREATMENTS:
        idx = [i for i, sid in enumerate(sample_ids)
               if samples[sid]["treatment"] == treatment]
        if len(idx) < 6:
            continue
        p = [probabilities[i] for i in idx]
        y = [1 if samples[sample_ids[i]]["label"] == "Fake" else 0 for i in idx]
        acc, f1 = _acc_f1(p, y, threshold)
        per_treatment[treatment] = {
            "n": len(idx), "auroc": _round(auroc(p, y)), "acc": round(acc, 4),
            "f1": round(f1, 4), "brier": round(brier(p, y), 4),
        }
    overall_acc, overall_f1 = _acc_f1(probabilities, labels, threshold)
    return {
        "n": len(labels),
        "auroc": _round(auroc(probabilities, labels)),
        "acc": round(overall_acc, 4), "f1": round(overall_f1, 4),
        "brier": round(brier(probabilities, labels), 4),
        "threshold": threshold,
        "per_treatment": per_treatment,
    }


def _round(value):
    return None if value is None or not np.isfinite(value) else round(float(value), 4)


def paired_delta(rows: List[dict], statistic, rounds: int = BOOTSTRAP_ROUNDS,
                 seed: int = 11) -> Optional[dict]:
    """
    Δ between two conditions, resampling **sources**.

    `rows` carries both conditions' probabilities for the same images, so the
    difference is paired — the same source appears in both sides of every
    resample, which is what makes the interval about the conditions rather
    than about the sample.
    """
    by_source: Dict[str, List[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], []).append(row)
    sources = sorted(by_source)
    if len(sources) < 4:
        return None
    point = statistic(rows)
    if point is None:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(rounds):
        picked = [rng.choice(sources) for _ in sources]
        sample = [row for source in picked for row in by_source[source]]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    if len(values) < rounds // 2:
        return None
    values.sort()
    return {"delta": round(float(point), 4),
            "ci": [round(values[int(0.025 * len(values))], 4),
                   round(values[int(0.975 * len(values)) - 1], 4)]}


# ---------------------------------------------------------------------------
# Fit and select
# ---------------------------------------------------------------------------

def build(group: str, conditions=CONDITIONS):
    sample_ids, table, samples = load_feature_table(group)
    return sample_ids, table, samples


def fit(group: str, conditions=CONDITIONS) -> dict:
    """Fit every condition on A and return models plus their A-group metrics."""
    sample_ids, table, samples = load_feature_table(group)
    labels = [1 if samples[sid]["label"] == "Fake" else 0 for sid in sample_ids]
    fitted: Dict[str, dict] = {}
    for name, features in conditions.items():
        X, medians = matrix(sample_ids, table, features)
        scaler = StandardScaler().fit(X)
        model = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X), labels)
        probabilities = model.predict_proba(scaler.transform(X))[:, 1].tolist()
        fitted[name] = {
            "features": features, "scaler": scaler, "model": model,
            "medians": medians, "probabilities": probabilities,
            "metrics": summarise(probabilities, labels, sample_ids, samples, DEFAULT_THRESHOLD),
            "coefficients": {f: round(float(c), 4)
                             for f, c in zip(features, model.coef_[0])},
        }
    return {"group": group, "sample_ids": sample_ids, "labels": labels,
            "samples": samples, "conditions": fitted}


def apply_fitted(state: dict, table: Dict[str, Dict[str, float]],
                 conditions=CONDITIONS) -> Dict[str, List[float]]:
    sample_ids = state["sample_ids"]
    out = {}
    for name, entry in state["conditions"].items():
        X, _ = matrix(sample_ids, table, entry["features"], entry["medians"])
        out[name] = entry["model"].predict_proba(entry["scaler"].transform(X))[:, 1].tolist()
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-select", action="store_true",
                        help="fit on A, select on B, write the freeze file")
    parser.add_argument("--validate-c", action="store_true",
                        help="read the sealed C scores; requires a frozen scheme")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if not (args.fit_select or args.validate_c):
        parser.error("choose --fit-select or --validate-c")

    if args.validate_c:
        validate_c()
        return

    fitted = fit("A")
    print("=== A 组（拟合）===")
    for name, entry in fitted["conditions"].items():
        m = entry["metrics"]
        print(f"  {name:16s} AUROC {m['auroc']}  Acc {m['acc']}  F1 {m['f1']}  "
              f"Brier {m['brier']}  系数 {entry['coefficients']}")

    b_ids, b_table, b_samples = load_feature_table("B")
    b_state = {"sample_ids": b_ids,
               "labels": [1 if b_samples[sid]["label"] == "Fake" else 0 for sid in b_ids],
               "conditions": {name: {"features": entry["features"],
                                     "scaler": entry["scaler"], "model": entry["model"],
                                     "medians": entry["medians"]}
                              for name, entry in fitted["conditions"].items()}}
    probabilities = apply_fitted(b_state, b_table)
    print("\n=== B 组（选择）===")
    selected = {}
    for name, probs in probabilities.items():
        threshold, f1 = best_threshold(probs, b_state["labels"])
        metrics = summarise(probs, b_state["labels"], b_ids, b_samples, threshold)
        selected[name] = {"metrics": metrics, "threshold": threshold}
        print(f"  {name:16s} AUROC {metrics['auroc']}  Acc {metrics['acc']}  "
              f"F1 {metrics['f1']}  Brier {metrics['brier']}  阈值 {threshold}")

    # Source-paired deltas between the conditions that matter.
    deltas = {}
    index = {sid: i for i, sid in enumerate(b_ids)}
    for left, right in (("existing", "probe"), ("existing+probe", "probe"),
                        ("existing+probe", "existing")):
        rows = [{"source_id": b_samples[sid]["source_id"],
                 "label": b_state["labels"][index[sid]],
                 "left": probabilities[left][index[sid]],
                 "right": probabilities[right][index[sid]]}
                for sid in b_ids]
        deltas[f"{left} - {right}"] = paired_delta(
            rows, lambda rs: auroc([r["left"] for r in rs], [r["label"] for r in rs]) -
            auroc([r["right"] for r in rs], [r["label"] for r in rs])) if rows else None
    print("\n按源配对的 ΔAUROC（B 组）:")
    for name, entry in deltas.items():
        if entry:
            print(f"  {name:26s} Δ {entry['delta']:+.4f}  CI {entry['ci']}")

    # The decision rule from Step 3, applied to the B-group numbers.
    fusion_delta = (deltas.get("existing+probe - probe") or {}).get("delta")
    if fusion_delta is None:
        verdict = "无法判定（B 组不足以比较）"
    elif fusion_delta >= FUSION_WINS_MARGIN:
        verdict = "融合稳定优于 PROBE 单独 → 保留多专家组合，继续研究调度"
    elif fusion_delta <= FUSION_LOSES_MARGIN:
        verdict = "融合无收益 → PROBE 作为主检测器，旧专家降为辅助线索"
    else:
        verdict = "收益在噪声范围内 → 按格式适用条件决定，不做全格式准入"

    freeze = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "split_hash": json.load(open(SPLIT_PATH, encoding="utf-8"))["hash"],
        "fitted_on": "A", "selected_on": "B",
        "conditions": {name: {"features": entry["features"],
                              "coefficients": entry["coefficients"],
                              "medians": entry["medians"],
                              "scaler_mean": [round(float(v), 6) for v in entry["scaler"].mean_],
                              "scaler_scale": [round(float(v), 6) for v in entry["scaler"].scale_],
                              "intercept": round(float(entry["model"].intercept_[0]), 6),
                              "threshold_b": selected[name]["threshold"],
                              "metrics_on_a": entry["metrics"],
                              "metrics_on_b": selected[name]["metrics"]}
                      for name, entry in fitted["conditions"].items()},
        "deltas_on_b": deltas,
        "decision_rule": {"fusion_wins_margin": FUSION_WINS_MARGIN,
                          "fusion_loses_margin": FUSION_LOSES_MARGIN},
        "verdict": verdict,
        "note": ("C 组在冻结前不得查看；--validate-c 会核对 split_hash 与冻结时间"),
    }
    output = args.output or FREEZE_PATH
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(freeze, handle, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"\n结论：{verdict}")
    print(f"冻结方案：{os.path.relpath(output, PROJECT_ROOT)}")


def validate_c() -> None:
    if not os.path.exists(FREEZE_PATH):
        raise SystemExit("尚未冻结方案（先跑 --fit-select）——C 组成绩在冻结前不得查看")
    with open(FREEZE_PATH, encoding="utf-8") as handle:
        freeze = json.load(handle)
    with open(SPLIT_PATH, encoding="utf-8") as handle:
        split_hash = json.load(handle)["hash"]
    if freeze["split_hash"] != split_hash:
        raise SystemExit("划分哈希与冻结方案不一致：分组变了，验证无效")

    sample_ids, table, samples = load_feature_table("C")
    labels = [1 if samples[sid]["label"] == "Fake" else 0 for sid in sample_ids]
    print(f"=== C 组（独立验证，方案冻结于 {freeze['generated_at']}）===")
    for name, entry in freeze["conditions"].items():
        X, _ = matrix(sample_ids, table, entry["features"], entry["medians"])
        scaler = _rebuild_scaler(entry)
        import numpy as _np
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.classes_ = _np.array([0, 1])
        model.coef_ = _np.array([[entry["coefficients"][f] for f in entry["features"]]])
        model.intercept_ = _np.array([entry["intercept"]])
        probabilities = model.predict_proba(scaler.transform(X))[:, 1].tolist()
        metrics = summarise(probabilities, labels, sample_ids, samples,
                            entry["threshold_b"])
        print(f"  {name:16s} AUROC {metrics['auroc']}  Acc {metrics['acc']}  "
              f"F1 {metrics['f1']}  Brier {metrics['brier']}（阈值 {entry['threshold_b']}，B 组选定）")
        for treatment, values in metrics["per_treatment"].items():
            print(f"      {treatment:10s} n={values['n']:3d} AUROC {values['auroc']} "
                  f"Acc {values['acc']} F1 {values['f1']}")
    # Paired deltas on C, computed *after* freezing and used to judge the
    # frozen scheme — never to choose one.  A verdict that does not replicate
    # here is the reason this group exists.
    index = {sid: i for i, sid in enumerate(sample_ids)}
    probabilities_by_condition = {}
    for name, entry in freeze["conditions"].items():
        X, _ = matrix(sample_ids, table, entry["features"], entry["medians"])
        scaler = _rebuild_scaler(entry)
        import numpy as _np
        model = LogisticRegression(max_iter=2000, C=1.0)
        model.classes_ = _np.array([0, 1])
        model.coef_ = _np.array([[entry["coefficients"][f] for f in entry["features"]]])
        model.intercept_ = _np.array([entry["intercept"]])
        probabilities_by_condition[name] = model.predict_proba(
            scaler.transform(X))[:, 1].tolist()

    print("\n按源配对的 ΔAUROC（C 组，冻结后只测量）:")
    for left, right in (("existing+probe", "probe"), ("existing", "probe")):
        if left not in probabilities_by_condition:
            continue
        rows = [{"source_id": samples[sid]["source_id"], "label": labels[index[sid]],
                 "left": probabilities_by_condition[left][index[sid]],
                 "right": probabilities_by_condition[right][index[sid]]}
                for sid in sample_ids]
        entry = paired_delta(rows, lambda rs: auroc([r["left"] for r in rs],
                                                    [r["label"] for r in rs]) -
                             auroc([r["right"] for r in rs], [r["label"] for r in rs]))
        if entry:
            print(f"  {left} - {right}: Δ {entry['delta']:+.4f}  CI {entry['ci']}")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_at": freeze["generated_at"], "split_hash": freeze["split_hash"],
        "group": "C", "n": len(labels),
        "conditions": {name: summarise(
            probabilities_by_condition[name], labels, sample_ids, samples,
            freeze["conditions"][name]["threshold_b"]) for name in probabilities_by_condition},
        "deltas": {f"{left} - {right}": paired_delta(
            [{"source_id": samples[sid]["source_id"], "label": labels[index[sid]],
              "left": probabilities_by_condition[left][index[sid]],
              "right": probabilities_by_condition[right][index[sid]]} for sid in sample_ids],
            lambda rs: auroc([r["left"] for r in rs], [r["label"] for r in rs]) -
            auroc([r["right"] for r in rs], [r["label"] for r in rs]))
            for left, right in (("existing+probe", "probe"), ("existing", "probe"))
            if left in probabilities_by_condition},
        "verdict_on_b": freeze["verdict"],
    }
    path = os.path.join(OUTPUT_DIR, "g7_validation_C.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1, allow_nan=False)
    print(f"\n冻结结论（在 B 组上得出）：{freeze['verdict']}")
    print("（若 C 组的 Δ 未复现该结论，以 C 组为准——这正是独立验证组的用途）")
    print(f"Report: {os.path.relpath(path, PROJECT_ROOT)}")


def _rebuild_scaler(entry):
    from sklearn.preprocessing import StandardScaler as _Scaler
    scaler = _Scaler()
    scaler.mean_ = np.array(entry["scaler_mean"])
    scaler.scale_ = np.array(entry["scaler_scale"])
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(entry["features"])
    return scaler


if __name__ == "__main__":
    main()
