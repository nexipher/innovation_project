#!/usr/bin/env python3
"""
Compare evaluated experts on the same images: overlap, and whether they add.

`g8_evaluate_experts.py` answers "how good is each expert"; this answers the
question the retained set actually turns on — **do they know different things?**
Two experts both at AUROC 0.7 can be redundant (one is a copy of the other) or
complementary (their mistakes are elsewhere), and only the second case earns a
place in the toolbox.

For each pair it reports the **mistake overlap** (how often both are wrong on
the same image, against what independence would predict) and a **naive average**
of their direction-corrected scores, so "combining them helps" is a number here
rather than an impression.  The average is deliberately the dumbest possible
combination: if a plain mean already beats both, complementarity is real; if it
does not, no amount of weighting will be trusted without a fitted scheme, and
fitting belongs on train/val — never on test.

Mock runs are refused (they are pipeline checks, not experts).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import PROJECT_ROOT
from utils.score_semantics import entry as semantics_entry


def load_run(path: str) -> Optional[dict]:
    """One evaluation run: its report (or None when it was a mock)."""
    report_path = os.path.join(path, "report.json")
    scores_path = os.path.join(path, "scores.jsonl")
    if not (os.path.exists(report_path) and os.path.exists(scores_path)):
        return None
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    if report.get("mock"):
        print(f"  跳过 Mock 运行：{os.path.basename(path)}")
        return None
    rows = {}
    with open(scores_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("raw_score") is not None:
                rows[row["variant_id"]] = row
    return {"label": report["expert"], "source_name": report["source_name"],
            "split": report["split"], "rows": rows, "path": path}


def decision_score(source_name: str, raw: float) -> float:
    """Direction-corrected score: higher always means 'more likely Fake'."""
    direction = semantics_entry(source_name)["raw_score"]["direction"]
    if direction == "higher_is_fake":
        return raw
    if direction == "higher_is_real":
        return 1.0 - raw
    raise ValueError(f"{source_name} 没有方向，不能参与对照")


def _auroc(y: np.ndarray, s: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    if len(set(y.tolist())) < 2:
        return None
    return float(roc_auc_score(y, s))


def compare(runs: List[dict], treatments: Optional[List[str]] = None) -> dict:
    shared = set.intersection(*[set(run["rows"]) for run in runs])
    if not shared:
        raise SystemExit("这些运行没有共同的图像（variant_id 不一致？）")
    ids = sorted(shared)
    labels = np.array([1 if runs[0]["rows"][i]["label"] == "Fake" else 0 for i in ids])
    treatments_of = {i: runs[0]["rows"][i]["treatment"] for i in ids}
    cells = treatments or sorted(set(treatments_of.values()))

    scores = {}
    for run in runs:
        raw = np.array([run["rows"][i]["raw_score"] for i in ids], dtype=float)
        scores[run["label"]] = np.array(
            [decision_score(run["source_name"], value) for value in raw])

    out = {"images": len(ids), "experts": [run["label"] for run in runs],
           "per_format": {}, "pairwise": [], "naive_average": None}
    for cell in cells:
        mask = np.array([treatments_of[i] == cell for i in ids])
        entry = {"n": int(mask.sum()), "auroc": {}}
        for name, values in scores.items():
            entry["auroc"][name] = _auroc(labels[mask], values[mask])
        out["per_format"][cell] = entry

    for i, first in enumerate(runs):
        for second in runs[i + 1:]:
            a, b = scores[first["label"]], scores[second["label"]]
            wrong_a, wrong_b = (a > 0.5) != (labels == 1), (b > 0.5) != (labels == 1)
            both = float((wrong_a & wrong_b).mean())
            expected = float(wrong_a.mean() * wrong_b.mean())
            out["pairwise"].append({
                "pair": [first["label"], second["label"]],
                "agreement": float(((a > 0.5) == (b > 0.5)).mean()),
                "both_wrong_rate": both,
                "independent_expectation": expected,
                "overlap_ratio": (both / expected) if expected else None,
                "note": ("重合低于独立预期 ⇒ 错误互补" if expected and both < expected
                         else "重合不低于独立预期 ⇒ 未显示互补"),
            })

    if len(runs) >= 2:
        mean_scores = np.mean([scores[run["label"]] for run in runs], axis=0)
        out["naive_average"] = {
            "auroc": _auroc(labels, mean_scores),
            "per_expert": {name: _auroc(labels, values) for name, values in scores.items()},
            "per_format": {},
        }
        for cell in cells:
            mask = np.array([treatments_of[i] == cell for i in ids])
            out["naive_average"]["per_format"][cell] = {
                "n": int(mask.sum()),
                "auroc_average": _auroc(labels[mask], mean_scores[mask]),
                **{name: _auroc(labels[mask], values[mask])
                   for name, values in scores.items()},
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", required=True,
                        help="评测目录（含 report.json 与 scores.jsonl），或 'latest' 自动取最新每个专家")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    paths = args.runs
    if paths == ["latest"]:
        latest = {}
        for path in sorted(glob.glob(os.path.join(PROJECT_ROOT, "sft_data", "g8_eval", "*"))):
            report = os.path.join(path, "report.json")
            if not os.path.exists(report):
                continue
            with open(report, encoding="utf-8") as handle:
                data = json.load(handle)
            if data.get("mock"):
                continue
            latest[data["expert"]] = path          # 目录名带时间戳，sorted 保证后者更新
        paths = sorted(latest.values())
        print(f"选用每个专家的最新运行：{[os.path.basename(p) for p in paths]}")

    runs = [run for run in (load_run(p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p))
                            for p in paths) if run]
    if len(runs) < 2:
        raise SystemExit("至少需要两个非 Mock 的评测运行才能谈互补性")
    splits = {run["split"] for run in runs}
    if len(splits) > 1:
        raise SystemExit(f"这些运行不在同一分区上：{splits}——不同分区不能混算")

    report = {"splits": sorted(splits), "runs": [run["path"] for run in runs],
              **compare(runs)}
    print(f"\n共同图像 {report['images']} 张｜专家 {report['experts']}")
    print("\n按格式的 AUROC：")
    header = "  " + "格式".ljust(10) + "".join(name.ljust(14) for name in report["experts"])
    print(header)
    for cell, stats in report["per_format"].items():
        line = "  " + f"{cell} (n={stats['n']})".ljust(10)
        for name in report["experts"]:
            value = stats["auroc"][name]
            line += (f"{value:.3f}" if value is not None else "—").ljust(14)
        print(line)
    print("\n错误重合：")
    for pair in report["pairwise"]:
        print(f"  {pair['pair'][0]} × {pair['pair'][1]}：一致 {pair['agreement']:.3f}｜"
              f"同时错 {pair['both_wrong_rate']:.3f}（独立预期 {pair['independent_expectation']:.3f}）"
              f"｜{pair['note']}")
    if report["naive_average"]:
        avg = report["naive_average"]
        print(f"\n朴素平均（方向已统一）：AUROC {avg['auroc']:.3f}｜各专家 "
              + "｜".join(f"{k} {v:.3f}" for k, v in avg["per_expert"].items()))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(f"\n报告：{os.path.relpath(args.out, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
