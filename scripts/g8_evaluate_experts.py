#!/usr/bin/env python3
"""
G8 unified offline expert evaluation (stage 3, plan.md §3c).

One entry for PROBE / noise / NPR (two checkpoints) / UnivFD, so four experts are
measured the same way on the same images with the same bookkeeping.  Every
measurement records what the user asked for:

  * **weights and code identity** — file, bytes, sha256, repo commit, and *how
    the file was loaded* (the NPR checkpoints need different conventions);
  * **input preprocessing** — name, detail and measurement scope, taken from the
    score-semantics registry rather than restated here;
  * **the raw score with its scale and direction** — and which field it came
    from: noise's `strength` and `raw_metric` point in opposite directions, so
    "the score" is not a well-formed phrase without the field name;
  * **status, elapsed time and errors** — a failed image is recorded as failed,
    never dropped or turned into a zero.

Partitions follow the user's rule: `val` is where experts, preprocessing and
thresholds are chosen; `test` is for the frozen scheme's final acceptance and
**refuses threshold search**.  Mock experts only exercise the pipeline: their
run is marked `mock: true` and kept out of the performance section.

Usage:
  python scripts/g8_evaluate_experts.py --expert mock-fixed --split val --limit 4   # CPU pipeline check
  python scripts/g8_evaluate_experts.py --expert noise --split val --limit 20       # CPU, real expert
  python scripts/g8_evaluate_experts.py --expert npr-a --split val --limit 20 --device cuda
  python scripts/g8_evaluate_experts.py --expert univfd --split val --limit 20 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from config import PROJECT_ROOT
from utils.score_semantics import entry as semantics_entry

SPLIT_PATH = os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "sft_data", "g8_eval")
FORMATS = ("png", "q70", "q85", "q95")
# The manifest's own vocabulary (`png` / `jpeg_q70`); the CLI takes the short
# form (`q70`) because that is what the evaluations are discussed in.
TREATMENT_ALIASES = {"png": "png", "q70": "jpeg_q70", "q85": "jpeg_q85",
                     "q95": "jpeg_q95", "native": "native", "source": "native"}


def _normalise_formats(formats) -> set:
    out = set()
    for name in formats:
        key = name.strip().lower()
        out.add(TREATMENT_ALIASES.get(key, key))
    return out

EXPERTS = {
    "probe": ("probe_dino_expert", False),
    "noise": ("noise_expert", False),
    "npr-a": ("npr_expert", False),
    "npr-b": ("npr_expert", False),
    "univfd": ("univfd_expert", False),
    "mock-fixed": ("probe_dino_expert", True),
    "mock-random": ("probe_dino_expert", True),
}


# ---------------------------------------------------------------------------
# Evaluators: one interface, whatever the expert is
# ---------------------------------------------------------------------------

class ExistingExpertEvaluator:
    """Wraps a project expert (PROBE, noise) that takes a BGR array."""

    def __init__(self, source_name: str, expert, field: str):
        self.source_name = source_name
        self._expert = expert
        self._field = field

    @property
    def name(self) -> str:
        return getattr(self._expert, "source_name", type(self._expert).__name__)

    def identity(self) -> dict:
        record = semantics_entry(self.source_name)
        provenance = getattr(self._expert, "provenance", None)
        weights = record["weights"]
        return {
            "expert": self.source_name,
            "weights_file": weights.get("file"),
            "weights_sha256": (provenance().sha256 if callable(provenance)
                               else weights.get("sha256")),
            "code_repo": record["code"]["repo"],
            "preprocess": record["preprocess"]["name"],
            "preprocess_detail": record["preprocess"]["detail"],
            "measurement_scope": record["preprocess"]["measurement_scope"],
            "licence": weights.get("licence"),
        }

    def score(self, image_path: str) -> float:
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        patch = np.ascontiguousarray(np.array(image)[:, :, ::-1])   # RGB → BGR uint8
        result = self._expert.analyze(patch)
        return float(getattr(result, self._field))


class MockEvaluator:
    """Deterministic stand-in: exercises the pipeline, never the results."""

    def __init__(self, source_name: str, kind: str, seed: int = 20261015):
        self.source_name = source_name
        self.kind = kind
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        return f"Mock({self.kind})"

    def identity(self) -> dict:
        return {"expert": self.source_name, "mock": True, "kind": self.kind,
                "weights_file": None, "weights_sha256": None,
                "preprocess": "无（Mock 不读像素）", "measurement_scope": "global",
                "licence": "不适用"}

    def score(self, image_path: str) -> float:
        if self.kind == "fixed":
            return 0.5
        return float(self._rng.random())


def build_evaluator(name: str, device: str):
    source_name, is_mock = EXPERTS[name]
    if is_mock:
        return MockEvaluator(source_name, "fixed" if name == "mock-fixed" else "random"), True
    field = semantics_entry(source_name)["raw_score"]["field"]
    if name == "probe":
        from experts.probe_dino import ProbeDinoExpert

        expert = ProbeDinoExpert(device=device)
        reason = expert.unavailable_reason()
        if reason:
            raise SystemExit(f"PROBE 不可用（{reason}）——设 PROBE_WEIGHTS 后再跑。")
        return ExistingExpertEvaluator(source_name, expert, field), False
    if name == "noise":
        from experts.noise import NoiseExpert

        return ExistingExpertEvaluator(source_name, NoiseExpert(), field), False
    if name in ("npr-a", "npr-b"):
        from experts.npr import NPREvaluator

        return NPREvaluator(checkpoint=name, device=device), False
    if name == "univfd":
        from experts.univfd import UnivFDEvaluator

        return UnivFDEvaluator(device=device), False
    raise SystemExit(f"未知专家：{name}")


# ---------------------------------------------------------------------------
# Images: the same encoding path the batch pipeline uses
# ---------------------------------------------------------------------------

def select_sources(split: str, limit: Optional[int], seed: int = 20261015) -> List[dict]:
    with open(SPLIT_PATH, encoding="utf-8") as handle:
        data = json.load(handle)
    chosen = []
    for key, record in data["sources"].items():
        if record.get("split") != split:
            continue
        chosen.append({"source_id": key, **record})
    chosen.sort(key=lambda item: item["source_id"])
    if limit:
        rng = np.random.default_rng(seed)
        picks = sorted(rng.choice(len(chosen), size=min(limit, len(chosen)), replace=False))
        chosen = [chosen[i] for i in picks]
    return chosen


def materialize_variants(sources: List[dict], out_dir: str, formats) -> List[dict]:
    """Source × format images, built by the project's canonical encoder."""
    from scripts.build_variants_g4 import build_variants

    with open(SPLIT_PATH, encoding="utf-8") as handle:
        split = json.load(handle)
    variants_dir = os.path.join(out_dir, "variants")
    manifest = build_variants([s["source_id"] for s in sources], split,
                              out_dir=variants_dir)
    wanted = _normalise_formats(formats)
    rows = []
    for key, record in manifest.items():
        treatment = record.get("treatment", "")
        if wanted and treatment not in wanted:
            continue
        # The manifest keys are project-relative (that is what the review
        # workstations link), so joining them onto the output directory looked
        # for the file one level too deep — every image failed to open.
        resolved = key if os.path.isabs(key) else os.path.join(PROJECT_ROOT, key)
        if not os.path.exists(resolved):
            resolved = os.path.join(variants_dir, os.path.basename(key))
        rows.append({"variant_id": record["variant_id"], "source_id": record["source_id"],
                     "label": record["label"], "treatment": treatment,
                     "path": resolved})
    rows.sort(key=lambda row: (row["source_id"], row["treatment"]))
    return rows


# ---------------------------------------------------------------------------
# Scoring and metrics
# ---------------------------------------------------------------------------

def score_all(evaluator, rows: List[dict]) -> List[dict]:
    out = []
    for row in rows:
        started = time.perf_counter()
        try:
            value = evaluator.score(row["path"])
            out.append({**row, "raw_score": value,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                        "status": "ok", "error": None})
        except Exception as exc:                      # recorded, never dropped
            out.append({**row, "raw_score": None,
                        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                        "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return out


def _auroc(labels: np.ndarray, scores: np.ndarray) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    if len(set(labels.tolist())) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def metrics_for(rows: List[dict], source_name: str, threshold: float) -> dict:
    """Metrics in the registry's direction, so 'high' always means the same thing."""
    direction = semantics_entry(source_name)["raw_score"]["direction"]
    ok = [row for row in rows if row["status"] == "ok" and row["raw_score"] is not None]
    if not ok:
        return {"n": 0, "note": "没有可用的测量"}
    y = np.array([1 if row["label"] == "Fake" else 0 for row in ok])
    raw = np.array([row["raw_score"] for row in ok], dtype=float)
    # 方向统一：把 registry 的方向翻成"越大越像 Fake"的判据分数
    if direction == "higher_is_fake":
        decision = raw
    elif direction == "higher_is_real":
        decision = 1.0 - raw
    else:
        decision = None
    out = {"n": len(ok), "direction": direction, "threshold_on_raw": threshold,
           "errors": sum(1 for row in rows if row["status"] == "error")}
    if decision is not None:
        out["auroc"] = _auroc(y, decision)
        out["accuracy@0.5"] = float(((decision > 0.5) == (y == 1)).mean())
        if direction == "higher_is_fake":
            out["accuracy@threshold"] = float(((raw > threshold) == (y == 1)).mean())
    out["per_format"] = {}
    for treatment in sorted({row["treatment"] for row in ok}):
        subset = [row for row in rows if row["status"] == "ok"
                  and row["treatment"] == treatment and row["raw_score"] is not None]
        ys = np.array([1 if row["label"] == "Fake" else 0 for row in subset])
        rs = np.array([row["raw_score"] for row in subset], dtype=float)
        dec = rs if direction == "higher_is_fake" else (1.0 - rs if direction == "higher_is_real" else None)
        out["per_format"][treatment] = {
            "n": len(subset),
            "auroc": _auroc(ys, dec) if dec is not None else None,
            "mean_fake_score": float(rs[ys == 1].mean()) if (ys == 1).any() else None,
            "mean_real_score": float(rs[ys == 0].mean()) if (ys == 0).any() else None,
        }
    out["mean_elapsed_ms"] = round(float(np.mean([row["elapsed_ms"] for row in ok])), 2)
    return out


# ---------------------------------------------------------------------------

def run(expert: str, split: str, limit: Optional[int], formats, device: str,
        out_dir: Optional[str] = None, threshold_search: bool = False) -> dict:
    if threshold_search and split != "val":
        raise SystemExit(
            "阈值只能在 val 上搜（阶段三口径）：test 分区用于方案冻结后的最终验收，"
            "在它上面搜阈值再报告准确率，等于把选择集当测试集。")
    source_name, is_mock = EXPERTS[expert]
    out_dir = out_dir or os.path.join(
        OUTPUT_ROOT, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{expert}_{split}")
    os.makedirs(out_dir, exist_ok=True)

    sources = select_sources(split, limit)
    if not sources:
        raise SystemExit(f"划分 {split} 里没有源图")
    rows = materialize_variants(sources, out_dir, formats)
    if not rows:
        raise SystemExit("没有可评测的变体（检查 --formats）")

    evaluator, is_mock = build_evaluator(expert, device)
    if hasattr(evaluator, "load"):
        evaluator.load()          # 先加载再取身份：加载方式本身是身份的一部分
    identity = evaluator.identity()
    scored = score_all(evaluator, rows)
    threshold = semantics_entry(source_name)["threshold"]["value"]
    metrics = metrics_for(scored, source_name, threshold)

    if threshold_search:
        from sklearn.metrics import roc_curve

        ok = [row for row in scored if row["status"] == "ok"]
        y = np.array([1 if row["label"] == "Fake" else 0 for row in ok])
        raw = np.array([row["raw_score"] for row in ok], dtype=float)
        direction = semantics_entry(source_name)["raw_score"]["direction"]
        dec = raw if direction == "higher_is_fake" else 1.0 - raw
        fpr, tpr, thr = roc_curve(y, dec)
        best = float(thr[int(np.argmax(tpr - fpr))])
        metrics["threshold_search_on_val"] = {"best_decision_threshold": best,
                                              "n": len(ok)}

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "expert": expert, "source_name": source_name, "mock": is_mock,
        "split": split, "formats": list(formats), "device": device,
        "sources": len(sources), "images": len(scored),
        "identity": identity,
        "semantics": {key: semantics_entry(source_name)["raw_score"][key]
                      for key in ("field", "range", "direction",
                                  "is_calibrated_probability")},
        "threshold_policy": {
            "value": threshold, "chosen_on": semantics_entry(source_name)["threshold"]["chosen_on"],
            "searched_on_val": bool(threshold_search),
        },
        "metrics": metrics,
        "failures": [{"variant_id": row["variant_id"], "error": row["error"]}
                     for row in scored if row["status"] == "error"],
    }
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "scores.jsonl"), "w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps({k: v for k, v in row.items() if k != "path"},
                                    ensure_ascii=False) + "\n")
    report["out_dir"] = os.path.relpath(out_dir, PROJECT_ROOT)

    if is_mock:
        report["note"] = ("这是 Mock 管线检查：**不计入任何性能结果**（用户 2026-10-15）。"
                          "真实性能只用非 mock 专家的报告。")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", required=True, choices=sorted(EXPERTS))
    parser.add_argument("--split", default="val", choices=("val", "test", "train"))
    parser.add_argument("--limit", type=int, default=None, help="抽多少源图")
    parser.add_argument("--formats", default=",".join(FORMATS),
                        help="变体格式：png,q70,q85,q95（也接受 manifest 的 jpeg_q70 写法）")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--threshold-search", action="store_true",
                        help="只在 val 上允许；test 分区会拒绝")
    args = parser.parse_args()

    formats = tuple(part.strip() for part in args.formats.split(",") if part.strip())
    report = run(args.expert, args.split, args.limit, formats, args.device,
                 args.out_dir, args.threshold_search)
    m = report["metrics"]
    print(f"专家 {report['expert']}（{report['source_name']}）｜{report['split']} 分区"
          f"｜{report['sources']} 源 / {report['images']} 张")
    print(f"  方向 {m.get('direction')}｜AUROC {m.get('auroc')}｜"
          f"acc@0.5 {m.get('accuracy@0.5')}｜错误 {m.get('errors')}｜"
          f"平均 {m.get('mean_elapsed_ms')} ms/图")
    for treatment, stats in (m.get("per_format") or {}).items():
        print(f"    {treatment:6} n={stats['n']:4} AUROC={stats['auroc']}"
              f"｜mean(Fake)={stats['mean_fake_score']}｜mean(Real)={stats['mean_real_score']}")
    if report["mock"]:
        print("  ⚠ Mock 管线检查：不计入性能结果")
    print(f"  报告：{report['out_dir']}/report.json")


if __name__ == "__main__":
    main()
