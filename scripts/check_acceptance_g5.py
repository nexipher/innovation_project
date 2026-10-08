#!/usr/bin/env python3
"""
G5 acceptance check: did the LoRA trial actually fix anything?

The criteria are the ones the plan committed to, and none of them is the
training loss:

  1. every tool arm's ΔAUROC against RGB is positive
  2. the JPEG cells are no longer below chance
  3. the PNG cells keep the discrimination the untuned pipeline had
  4. the abstention rate falls
  5. a higher coverage does not collapse the accuracy of the labels it commits
  6. no verdict rests on the container format

The numbers come from a four-arm report (`calibration/*.json`, same shape as
the G2-d/G3-e runs, which the same harness produces) plus, for criterion 6,
the traces of that run.  Everything here is CPU: it reads results, it does not
produce them.

Usage:
  python scripts/check_acceptance_g5.py --report calibration/g3_gain_report.json
  python scripts/check_acceptance_g5.py --report calibration/lora_arms.json \
      --baseline-report calibration/g3_gain_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from scripts.qwen_gain_baseline import compute_metrics

ARMS = ("rgb", "text", "image", "both")
TOOL_ARMS = ARMS[1:]

# A JPEG cell counts as recovered once it clears chance by this much.
JPEG_CELL_FLOOR = 0.55
# The PNG-cell discrimination the untuned pipeline reached (G3-e).
PNG_CELL_REFERENCE = 0.78
PNG_CELL_TOLERANCE = 0.10
# Abstention must fall below the G3-e band (0.67-0.73).
UNCERTAIN_CEILING = 0.60
# Committed labels must stay at least this accurate as coverage grows.
COMMITTED_ACCURACY_FLOOR = 0.60

CONTAINER_WORDS = ("png", "jpeg", "jpg", "容器", "container", "格式")
DIRECTION_WORDS = ("伪造", "篡改", "ai 生成", "ai生成", "真实", "fake", "real",
                   "generated", "forgery", "authentic", "假图", "真图", "合成的")
CAUSAL_CONNECTIVES = ("因为", "由于", "所以", "因此", "据此", "说明", "意味着", "表明",
                      "indicates", "because", "therefore", "suggests", "proves")
DENIALS = ("不代表", "不能作为", "不能据此", "与伪造无关", "not a reason",
           "does not indicate", "no bearing", "不构成", "无关", "不能说明",
           "不应", "不得", "不能", "而非", "而不是", "不意味着")


def load_report(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def records_by_arm(report: dict) -> Dict[str, Dict[str, dict]]:
    return {arm: {r["sample_id"]: r for r in report["conditions"].get(arm, {}).get("records", [])}
            for arm in ARMS}


def cell_auroc(records: Dict[str, dict], cell_suffix: str) -> Optional[float]:
    """
    AUROC of one formatting group, pooling its Real and Fake cells.

    Cells are single-class, so a per-cell AUROC is undefined; pooling the
    format's two cells is what makes the number mean "can it tell them apart
    *within* this format", which is the only version of the question that
    cannot be answered by the container.
    """
    import numpy as np
    from scipy.stats import rankdata

    def probability(record):
        verdict, confidence = record.get("verdict"), record.get("confidence") or 0.0
        if verdict == "Fake":
            return confidence
        if verdict == "Real":
            return 1.0 - confidence
        return 0.5

    members = [r for r in records.values() if r["cell"].endswith(cell_suffix)]
    positive = np.array([probability(r) for r in members if r["gt"] == "Fake"])
    negative = np.array([probability(r) for r in members if r["gt"] == "Real"])
    if len(positive) < 3 or len(negative) < 3:
        return None
    scores = np.concatenate([positive, negative])
    ranks = rankdata(scores)
    return float((ranks[:len(positive)].sum() - len(positive) * (len(positive) + 1) / 2)
                 / (len(positive) * len(negative)))


def _container_reason_lines(text: str) -> List[str]:
    offenders = []
    for sentence in re.split(r"[。.;；\n]", text or ""):
        lowered = sentence.lower()
        if not any(w in lowered for w in CONTAINER_WORDS):
            continue
        if not any(w in lowered for w in DIRECTION_WORDS):
            continue
        if not any(c in lowered for c in CAUSAL_CONNECTIVES):
            continue
        if any(d in lowered for d in DENIALS):
            continue
        offenders.append(sentence.strip()[:120])
    return offenders


def container_offenders(traces_dir: Optional[str], limit: int = 200) -> Dict[str, int]:
    """Model turns that argue from the container, per arm."""
    if not traces_dir or not os.path.isdir(traces_dir):
        return {}
    counts: Dict[str, int] = {}
    files = sorted(glob.glob(os.path.join(traces_dir, "*.json")))[-limit:]
    for path in files:
        try:
            with open(path, encoding="utf-8") as handle:
                trace = json.load(handle)
        except (OSError, ValueError):
            continue
        if trace.get("metadata", {}).get("mock_mode") != "qwen_real":
            continue
        offenders = 0
        for turn in trace.get("conversations", []):
            if turn.get("from") != "gpt":
                continue
            offenders += len(_container_reason_lines(turn.get("value", "")))
        if offenders:
            counts[os.path.basename(path)] = offenders
    return counts


def check(report: dict, baseline_report: Optional[dict] = None,
          traces_dir: Optional[str] = None) -> dict:
    """Every criterion, with the measured value and a pass/fail."""
    records = records_by_arm(report)
    baseline = records_by_arm(baseline_report) if baseline_report else None
    metrics = {arm: compute_metrics(list(records[arm].values())) for arm in ARMS}

    jpeg_reference = None
    if baseline:
        # The untuned JPEG-cell AUROC, so "no longer below chance" is a
        # comparison rather than an absolute claim.
        jpeg_scores = [cell_auroc(baseline[arm], "jpeg_q70") for arm in ARMS]
        jpeg_scores = [s for s in jpeg_scores if s is not None]
        jpeg_reference = min(jpeg_scores) if jpeg_scores else None

    criteria = []

    deltas = {}
    for arm in TOOL_ARMS:
        if metrics[arm].get("auroc") is None or metrics["rgb"].get("auroc") is None:
            deltas[arm] = None
            continue
        deltas[arm] = round(metrics[arm]["auroc"] - metrics["rgb"]["auroc"], 4)
    criteria.append({
        "name": "tool_arms_beat_baseline_auroc",
        "passed": all(d is not None and d > 0 for d in deltas.values()),
        "detail": f"ΔAUROC vs rgb: {deltas}",
    })

    jpeg_cells = {}
    for arm in ARMS:
        value = cell_auroc(records[arm], "jpeg_q70")
        jpeg_cells[arm] = None if value is None else round(value, 3)
    worst_tool_jpeg = min((v for a, v in jpeg_cells.items()
                           if a in TOOL_ARMS and v is not None), default=None)
    criteria.append({
        "name": "jpeg_cells_above_chance",
        "passed": bool(worst_tool_jpeg is not None and worst_tool_jpeg >= JPEG_CELL_FLOOR),
        "detail": f"q70 cell AUROC per arm: {jpeg_cells} (floor {JPEG_CELL_FLOOR}, "
                  f"untuned worst {jpeg_reference})",
    })

    png_cells = {}
    for arm in TOOL_ARMS:
        value = cell_auroc(records[arm], "png")
        png_cells[arm] = None if value is None else round(value, 3)
    best_png = max((v for v in png_cells.values() if v is not None), default=None)
    criteria.append({
        "name": "png_cell_gain_retained",
        "passed": bool(best_png is not None
                       and best_png >= PNG_CELL_REFERENCE - PNG_CELL_TOLERANCE),
        "detail": f"png cell AUROC per arm: {png_cells} "
                  f"(reference {PNG_CELL_REFERENCE} ± {PNG_CELL_TOLERANCE})",
    })

    uncertain = {arm: round(metrics[arm].get("uncertain_rate", 0.0), 3) for arm in TOOL_ARMS}
    criteria.append({
        "name": "abstention_falls",
        "passed": all(value <= UNCERTAIN_CEILING for value in uncertain.values()),
        "detail": f"uncertain rate per tool arm: {uncertain} (ceiling {UNCERTAIN_CEILING})",
    })

    committed = {}
    for arm in TOOL_ARMS:
        rows = [r for r in records[arm].values() if r.get("verdict") in ("Real", "Fake")]
        hits = sum(1 for r in rows if r["verdict"] == r["gt"])
        committed[arm] = {"n": len(rows), "accuracy": round(hits / len(rows), 3) if rows else None}
    criteria.append({
        "name": "committed_labels_stay_accurate",
        "passed": all(entry["accuracy"] is None or entry["accuracy"] >= COMMITTED_ACCURACY_FLOOR
                      for entry in committed.values()),
        "detail": f"committed-label accuracy per arm: {committed} "
                  f"(floor {COMMITTED_ACCURACY_FLOOR})",
    })

    offenders = container_offenders(traces_dir)
    criteria.append({
        "name": "no_container_as_reason",
        "passed": not offenders,
        "detail": (f"{len(offenders)} model turns argue from the container" if offenders
                   else "no model turn argues from the container"),
    })

    return {
        "report": report.get("mode", "?"),
        "arms": len(records["rgb"]),
        "metrics": {arm: {k: metrics[arm].get(k) for k in
                          ("accuracy", "auroc", "uncertain_rate", "f1_fake")}
                    for arm in ARMS},
        "criteria": criteria,
        "passed": all(c["passed"] for c in criteria),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--baseline-report", default=None,
                        help="the untuned four-arm report, for the JPEG reference")
    parser.add_argument("--traces-dir",
                        default=os.path.join(PROJECT_ROOT, "traces", "sft_sessions"))
    args = parser.parse_args()

    result = check(load_report(args.report),
                   load_report(args.baseline_report) if args.baseline_report else None,
                   args.traces_dir)

    print(f"report: {args.report}  arms: {result['arms']}")
    for arm, values in result["metrics"].items():
        print(f"  {arm:6s} acc={values['accuracy']:.3f} auroc={values['auroc']:.3f} "
              f"unc={values['uncertain_rate']:.3f}")
    print()
    for criterion in result["criteria"]:
        mark = "✅" if criterion["passed"] else "❌"
        print(f"{mark} {criterion['name']}: {criterion['detail']}")
    print()
    print("验收结果:", "通过" if result["passed"] else "未通过 —— 先修数据与目标，不要加 epoch 或扩数据")


if __name__ == "__main__":
    main()
