"""Report the existing MNW/DIV2K q95 measurements without rerunning models.

The scores were produced on 2026-10-26 under `sft_data/g8_external_eval/`.
This script filters the q95 condition only, pairs each newer-generator group
with the shared DIV2K real images, and writes a standalone report.  It never
mixes GenImage with MNW and never fits a threshold.
"""

from __future__ import annotations

import collections
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.g8_external_eval import (
    bootstrap_auroc,
    expert_orientation,
    predicts_fake,
)

ROOT = Path("/root/autodl-tmp/innovation_project")
RUNS = {
    "probe": ROOT / "sft_data/g8_external_eval/dev1_probe_20261026_2132",
    "spai": ROOT / "sft_data/g8_external_eval/dev1_spai_20261026_2145",
}
OUT = ROOT / "sft_data/g8_external_eval/mnw_q95_existing_20261102.json"
OUT_MD = ROOT / "sft_data/g8_external_eval/mnw_q95_existing_20261102.md"
REAL_GROUP = "DIV2K_valid_HR"
FAKE_GROUPS = ("Adobe_Firefly_v4", "ByteDance_Seedream_v4", "Google_Imagen4",
               "Midjourney_v7")


def load_rows(path: Path):
    rows = []
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        if row.get("condition") == "jpeg_q95":
            rows.append(row)
    return rows


def distribution(values):
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return {
        "n": len(values),
        "mean": round(float(a.mean()), 6),
        "std": round(float(a.std(ddof=0)), 6),
        "min": round(float(a.min()), 6),
        "p05": round(float(np.percentile(a, 5)), 6),
        "p25": round(float(np.percentile(a, 25)), 6),
        "p50": round(float(np.percentile(a, 50)), 6),
        "p75": round(float(np.percentile(a, 75)), 6),
        "p95": round(float(np.percentile(a, 95)), 6),
        "max": round(float(a.max()), 6),
    }


def metrics(fake_rows, real_rows, expert):
    rows = fake_rows + real_rows
    measured = [row for row in rows
                if row.get("status", "measured") == "measured"
                and row.get("value") is not None]
    failures = len(rows) - len(measured)
    if not measured:
        return {"n": len(rows), "n_measured": 0, "failures": failures,
                "auroc": None, "tpr": None, "tnr": None,
                "balanced_accuracy": None, "score_distribution": None,
                "errors": []}
    scores = [float(row["oriented"]) for row in measured]
    labels = [1 if row["label"] == "Fake" else 0 for row in measured]
    fake = [float(row["oriented"]) for row in measured if row["label"] == "Fake"]
    real = [float(row["oriented"]) for row in measured if row["label"] == "Real"]
    tpr = (sum(1 for row in measured
               if row["label"] == "Fake" and predicts_fake(
                   row["value"], row["direction"], row["threshold"]))
           / len(fake)) if fake else None
    tnr = (sum(1 for row in measured
               if row["label"] == "Real" and not predicts_fake(
                   row["value"], row["direction"], row["threshold"]))
           / len(real)) if real else None
    errors = []
    for row in measured:
        predicted = "Fake" if predicts_fake(
            row["value"], row["direction"], row["threshold"]) else "Real"
        if predicted != row["label"]:
            errors.append({
                "variant_id": row.get("variant_id"),
                "source_id": row.get("source_id"),
                "label": row["label"],
                "predicted": predicted,
                "score": round(float(row["value"]), 6),
            })
    elapsed = [float(row["elapsed_ms"]) for row in measured
               if row.get("elapsed_ms") is not None]
    return {
        "n": len(rows), "n_measured": len(measured),
        "n_fake": len(fake), "n_real": len(real), "failures": failures,
        "auroc": bootstrap_auroc(scores, labels),
        "tpr": round(tpr, 6) if tpr is not None else None,
        "tnr": round(tnr, 6) if tnr is not None else None,
        "balanced_accuracy": (round((tpr + tnr) / 2, 6)
                              if tpr is not None and tnr is not None else None),
        "threshold": 0.5,
        "score_distribution": {"fake": distribution(fake),
                               "real": distribution(real)},
        "elapsed_ms": ({
            "n": len(elapsed), "mean": round(statistics.mean(elapsed), 1),
            "median": round(statistics.median(elapsed), 1),
            "min": round(min(elapsed), 1), "max": round(max(elapsed), 1),
        } if elapsed else None),
        "errors": errors,
    }


def main():
    report = {
        "schema": "mnw_q95_existing/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "Existing 2026-10-26 MNW/DIV2K measurements; q95 only; no rerun.",
        "rules": {
            "only_q95": True,
            "no_threshold_fitting": True,
            "per_generator_only": True,
            "real_images_shared": True,
            "no_genimage_mixing": True,
        },
        "experts": {},
        "groups": {},
    }
    for expert, directory in RUNS.items():
        scores_path = directory / "scores.jsonl"
        run_report = json.load(open(directory / "report.json", encoding="utf-8"))
        rows = load_rows(scores_path)
        report["experts"][expert] = {
            "scores": str(scores_path.relative_to(ROOT)),
            "identity": run_report.get("experts", {}).get(expert),
            "run": run_report.get("run"),
        }
        real = [row for row in rows
                if row["group"] == REAL_GROUP and row["label"] == "Real"]
        for group in FAKE_GROUPS:
            fake = [row for row in rows
                    if row["group"] == group and row["label"] == "Fake"]
            report["groups"].setdefault(group, {})[expert] = metrics(
                fake, real, expert)
        report["groups"].setdefault("Real_only", {})[expert] = metrics(
            [], real, expert)

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    lines = [
        "# MNW/DIV2K q95 既有测量（单独报告）",
        "",
        f"生成时间：{report['created_at']}。来源：2026-10-26 既有 scores；"
        "**只取 q95，不重跑，不混入 GenImage，不拟合阈值**。",
        "",
        "| 生成器 | 专家 | 假图 n | 真图 n | AUROC | TPR | TNR | 平衡准确率 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for group in FAKE_GROUPS:
        for expert in RUNS:
            m = report["groups"][group][expert]
            auroc = m["auroc"]["auroc"] if m["auroc"] else None
            lines.append(
                f"| {group} | {expert} | {m['n_fake']} | {m['n_real']} | "
                f"{'—' if auroc is None else f'{auroc:.3f}'} | "
                f"{'—' if m['tpr'] is None else f'{m['tpr']:.3f}'} | "
                f"{'—' if m['tnr'] is None else f'{m['tnr']:.3f}'} | "
                f"{'—' if m['balanced_accuracy'] is None else f'{m['balanced_accuracy']:.3f}'} |")
    lines += [
        "",
        "四个生成器组共用同一批 DIV2K Real；组间不独立，因此不报告合并 AUROC。"
        "完整分数分布、错误样本、失败和耗时见同名 JSON。",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)),
                      "groups": list(report["groups"])}, ensure_ascii=False,
                     indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
