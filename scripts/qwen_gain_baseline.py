#!/usr/bin/env python3
"""
G2-d: four-condition gain comparison (RGB / +text / +visual / +both).

Measures whether expert evidence adds anything over the raw model on the
calibration set, and which channel (text token vs visual artifact) carries
the signal:

  rgb    no-tool baseline — BASELINE_SYSTEM_PROMPT, evidence_injection="none"
  text   Evidence Token JSON only                   (G1 behaviour)
  image  region crop + artifacts, neutral marker    (no numbers shown)
  both   Evidence Token JSON + images               (G2-c default)

The Qwen client loads once and is reused across samples. Running the real
comparison requires a GPU (see agent.md §3.2); `--dry-run` validates the
whole harness with MockMLLMClient on CPU.

Note on the dry run: MockMLLMClient derives its verdict from the Evidence
Token JSON in the conversation, so the `rgb` and `image` arms — which send
the model no numbers by construction — come back `Uncertain` and score at
chance.  That is the mock reflecting the injection mode, not a measurement:
the dry run only proves the plumbing, the science needs the GPU.

Usage:
  python scripts/qwen_gain_baseline.py --dry-run --per-cell 2      # CPU smoke
  python scripts/qwen_gain_baseline.py --per-cell 15               # GPU (authorized)
  python scripts/qwen_gain_baseline.py --conditions rgb both
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from mllm.message_builder import BASELINE_SYSTEM_PROMPT
from mllm.mock_client import MockMLLMClient
from state_machine.controller import ForensicStateMachine
from utils import config_fingerprint
from utils.logger import SessionLogger

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "calibration", "set", "manifest.json")
REPORT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g2_gain_report.json")
DRY_RUN_REPORT_PATH = os.path.join(
    PROJECT_ROOT, "calibration", "g2_gain_report_dry_run.json"
)
# Mock sessions must not land among the real ones that finalize_sft_data.py
# scans by filename (scripts/finalize_sft_data.py).
DRY_RUN_SESSIONS_DIR = os.path.join(PROJECT_ROOT, "traces", "dry_run_sessions")


def report_path(dry_run: bool, override: Optional[str] = None) -> str:
    """
    Where this invocation writes its report.

    Dry-run output is kept apart so mock numbers can never overwrite the
    report of an authorized GPU run; `override` keeps later comparisons (for
    example a rerun after a change to the experts) from overwriting the run
    they are being compared against.
    """
    if override:
        return override if os.path.isabs(override) else os.path.join(PROJECT_ROOT, override)
    return DRY_RUN_REPORT_PATH if dry_run else REPORT_PATH

# Cells used for the comparison: format-matched PNG plus the compression-matched
# JPEG qualities (any confounded native-cell signal would inflate the numbers).
COMPARISON_CELLS = ("real_png", "fake_png", "real_jpeg_q95", "fake_jpeg_q95",
                    "real_jpeg_q85", "fake_jpeg_q85", "real_jpeg_q70", "fake_jpeg_q70")

CONDITIONS = {
    "rgb": {"injection": "none", "client": "baseline"},
    "text": {"injection": "text", "client": "forensic"},
    "image": {"injection": "image", "client": "forensic"},
    "both": {"injection": "text+image", "client": "forensic"},
}


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def select_samples(manifest: dict, per_class_per_cell: int) -> List[dict]:
    """
    Stratified selection: equal Real/Fake per format-matched cell.

    Each cell is single-class (`real_png` vs `fake_png`), so taking N samples
    per cell yields N Real and N Fake for every format/quality pairing — the
    comparison cannot be won by class-format correlation.
    """
    by_cell: Dict[str, Dict[str, List[dict]]] = {}
    for sample in manifest["samples"]:
        if sample["cell"] not in COMPARISON_CELLS:
            continue
        by_cell.setdefault(sample["cell"], {"Real": [], "Fake": []})
        by_cell[sample["cell"]][sample["label"]].append(sample)

    selected: List[dict] = []
    for cell in COMPARISON_CELLS:
        groups = by_cell.get(cell, {})
        for label in ("Real", "Fake"):
            selected.extend(groups.get(label, [])[:per_class_per_cell])
    return selected


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _pseudo_probability(record: dict) -> float:
    """P(Fake) estimate from the verdict and its confidence."""
    verdict = record.get("verdict")
    confidence = float(record.get("confidence") or 0.0)
    if verdict == "Fake":
        return confidence
    if verdict == "Real":
        return 1.0 - confidence
    return 0.5  # Uncertain carries no directional information


def _auroc(records: List[dict]) -> Optional[float]:
    positive = [_pseudo_probability(r) for r in records if r["gt"] == "Fake"]
    negative = [_pseudo_probability(r) for r in records if r["gt"] == "Real"]
    if len(positive) < 3 or len(negative) < 3:
        return None
    scores = np.concatenate([positive, negative])
    ranks = rankdata(scores)
    n_pos = len(positive)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * len(negative)))


def _ece(records: List[dict], bins: int = 10) -> Optional[float]:
    """Expected calibration error of the pseudo-probability, 10 equal-width bins."""
    if len(records) < 10:
        return None
    probabilities = np.array([_pseudo_probability(r) for r in records])
    outcomes = np.array([1.0 if r["gt"] == "Fake" else 0.0 for r in records])
    total = len(records)
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        mask = (probabilities >= low) & (probabilities < high)
        if index == bins - 1:
            mask = (probabilities >= low) & (probabilities <= high)
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = outcomes[mask].mean()
        confidence = probabilities[mask].mean()
        ece += (count / total) * abs(accuracy - confidence)
    return float(ece)


def compute_metrics(records: List[dict]) -> dict:
    """Accuracy / F1 / AUROC / ECE / Uncertain rate / cost for one condition."""
    if not records:
        return {"n": 0}

    correct = sum(1 for r in records if r["verdict"] == r["gt"])
    tp = sum(1 for r in records if r["gt"] == "Fake" and r["verdict"] == "Fake")
    fp = sum(1 for r in records if r["gt"] == "Real" and r["verdict"] == "Fake")
    fn = sum(1 for r in records if r["gt"] == "Fake" and r["verdict"] != "Fake")
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    per_cell: Dict[str, Dict[str, int]] = {}
    for record in records:
        bucket = per_cell.setdefault(record["cell"], {"n": 0, "correct": 0})
        bucket["n"] += 1
        bucket["correct"] += int(record["verdict"] == record["gt"])

    return {
        "n": len(records),
        "accuracy": correct / len(records),
        "f1_fake": f1,
        "auroc": _auroc(records),
        "ece": _ece(records),
        "uncertain_rate": sum(1 for r in records if r["verdict"] == "Uncertain") / len(records),
        "avg_model_turns": float(np.mean([r.get("model_turns", 0) for r in records])),
        "avg_expert_calls": float(np.mean([r.get("expert_calls", 0) for r in records])),
        "per_cell_accuracy": {
            cell: bucket["correct"] / bucket["n"] for cell, bucket in sorted(per_cell.items())
        },
    }


# ---------------------------------------------------------------------------
# Condition runner
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    samples: List[dict],
    client_factory: Callable[[str], Any],
    progress: bool = True,
    sft_dir: Optional[str] = None,
    on_record: Optional[Callable[[dict], None]] = None,
) -> List[dict]:
    """
    Run every sample under one condition; return per-sample records.

    The MLLM client and the expert toolkit are built once per condition and
    reused across samples: `QwenVLClient` loads 16.6 GB of weights on first
    use, so constructing one per sample would reload the model for every
    image.  The state machine is rebuilt per sample, which is what resets the
    session (`run()` also calls `mllm.reset()`).

    Args:
        condition: One of CONDITIONS.
        samples: Manifest sample dicts.
        client_factory: Called with "baseline" | "forensic" -> MLLM client.
        sft_dir: Trace directory override (dry runs keep mock sessions apart).
        on_record: Called with each record as it is produced, so a caller can
            persist progress — an hours-long GPU run must survive a kill.
    """
    spec = CONDITIONS[condition]
    records: List[dict] = []
    client = client_factory(spec["client"])
    experts = _build_experts()

    for index, sample in enumerate(samples, 1):
        image_path = os.path.join(PROJECT_ROOT, sample["path"])
        fsm = ForensicStateMachine(
            client,
            experts,
            logger=SessionLogger(sft_dir=sft_dir) if sft_dir else None,
            evidence_injection=spec["injection"],
        )
        started = time.perf_counter()
        result = fsm.run(image_path, ground_truth=sample["label"])
        elapsed = time.perf_counter() - started

        verdict = result["final_verdict"]
        record = {
            "sample_id": sample["sample_id"],
            "cell": sample["cell"],
            "gt": sample["label"],
            "verdict": verdict.get("verdict"),
            "confidence": verdict.get("confidence"),
            "model_turns": result.get("model_turn_count", 0),
            "expert_calls": result.get("expert_call_count", 0),
            "elapsed_s": round(elapsed, 2),
        }
        records.append(record)
        if on_record is not None:
            on_record(record)
        if progress and index % 20 == 0:
            print(f"    [{index}/{len(samples)}] {condition}")

    return records


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_completed(path: str, mode: str, per_cell: int,
                   fingerprint: Optional[dict] = None) -> Dict[str, List[dict]]:
    """
    Records per condition from an earlier run of the *same experiment*.

    Returns {} when the file is absent, unreadable, produced by a different
    mode or sample size, or — since G3-d — recorded under a different
    configuration fingerprint.  The fingerprint covers the expert registry and
    their polarities, the prompts, the reliability table and the rectifier
    version, so a run made after a change can no longer resume the numbers it
    was supposed to replace (which used to require remembering --fresh).
    """
    try:
        with open(path, encoding="utf-8") as handle:
            previous = json.load(handle)
    except (OSError, ValueError):
        return {}
    if previous.get("mode") != mode or previous.get("per_cell") != per_cell:
        return {}
    if fingerprint is not None:
        stored = previous.get("config_fingerprint")
        if not stored or stored.get("digest") != fingerprint.get("digest"):
            changed = config_fingerprint.differences(stored, fingerprint)
            print(f"  configuration changed ({', '.join(changed)}) — cold start")
            return {}
    return {
        condition: entry.get("records", [])
        for condition, entry in (previous.get("conditions") or {}).items()
    }


def pending_samples(samples: List[dict], done: List[dict]) -> List[dict]:
    """Samples not yet recorded for this condition."""
    finished = {record["sample_id"] for record in done}
    return [sample for sample in samples if sample["sample_id"] not in finished]


def initial_completed(output_path: str, mode: str, per_cell: int,
                      fresh: bool,
                      fingerprint: Optional[dict] = None) -> Dict[str, List[dict]]:
    """Resume state for this invocation (nothing when --fresh)."""
    if fresh:
        return {}
    return load_completed(output_path, mode, per_cell, fingerprint)


def _write_report(path: str, report: dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def _build_experts() -> Dict[str, Any]:
    from experts.frequency_v2 import FrequencyExpertV2
    from experts.jpeg import JPEGExpert
    from experts.noise import NoiseExpert
    # G2 evidence favours v2; the gain comparison measures the toolkit the
    # pipeline will actually ship, not the disabled v1.
    return {
        "frequency_expert_v2": FrequencyExpertV2(),
        "noise_expert": NoiseExpert(),
        "jpeg_expert": JPEGExpert(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-cell", type=int, default=15,
                        help="Real and Fake samples per comparison cell")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS))
    parser.add_argument("--dry-run", action="store_true",
                        help="Use MockMLLMClient (CPU plumbing validation only)")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore an existing report and start over "
                             "(default: resume the samples already recorded)")
    parser.add_argument("--output", default=None,
                        help="Report path, relative to the project root "
                             "(default: the fixed GPU or dry-run report)")
    args = parser.parse_args()

    unknown = [c for c in args.conditions if c not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown conditions: {unknown}")

    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    samples = select_samples(manifest, args.per_cell)
    print(f"Samples: {len(samples)} "
          f"({args.per_cell} Real + {args.per_cell} Fake per format/quality cell, "
          f"{len(set(s['cell'] for s in samples))} cells)")

    if args.dry_run:
        def client_factory(_variant):
            return MockMLLMClient(mode="two_calls", seed=42)
        mode = "dry_run"
    else:
        if not _cuda_available():
            raise SystemExit(
                "CUDA not available. The real comparison requires the RTX 4090 "
                "(agent.md §3.2: obtain user authorization first). "
                "Use --dry-run for CPU plumbing validation."
            )
        from mllm.qwen_client import QwenVLClient

        def client_factory(variant):
            prompt = BASELINE_SYSTEM_PROMPT if variant == "baseline" else None
            return QwenVLClient(system_prompt=prompt)
        mode = "gpu"

    output_path = report_path(args.dry_run, args.output)
    fingerprint = config_fingerprint.compute(_build_experts())
    completed = initial_completed(output_path, mode, args.per_cell, args.fresh,
                                  fingerprint)
    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "per_cell": args.per_cell,
        "samples": len(samples),
        "config_fingerprint": fingerprint,
        "conditions": {
            condition: {"metrics": compute_metrics(records), "records": records}
            for condition, records in completed.items()
        },
    }
    if completed:
        done = sum(len(records) for records in completed.values())
        print(f"Resuming: {done} records already on disk")

    sft_dir = DRY_RUN_SESSIONS_DIR if args.dry_run else None
    for condition in args.conditions:
        entry = report["conditions"].setdefault(
            condition, {"metrics": {}, "records": []}
        )
        todo = pending_samples(samples, entry["records"])
        if not todo:
            print(f"\n=== Condition: {condition} — already complete, skipping ===")
            continue
        print(f"\n=== Condition: {condition} ({len(todo)} samples) ===")

        def on_record(record, entry=entry):
            # Persist after every sample: an interrupted GPU run resumes
            # instead of starting over.
            entry["records"].append(record)
            entry["metrics"] = compute_metrics(entry["records"])
            report["generated_at"] = datetime.now().isoformat(timespec="seconds")
            _write_report(output_path, report)

        run_condition(condition, todo, client_factory,
                      sft_dir=sft_dir, on_record=on_record)

    _write_report(output_path, report)
    print(f"\nReport: {output_path}")

    # ---- console summary ---------------------------------------------
    print(f"\n{'condition':10s} {'acc':>6s} {'f1':>6s} {'auroc':>6s} {'ece':>6s} "
          f"{'unc':>6s} {'turns':>6s} {'calls':>6s}")
    for condition, entry in report["conditions"].items():
        metrics = entry["metrics"]
        def fmt(key, spec=".3f"):
            value = metrics.get(key)
            return f"{value:{spec}}" if value is not None else "  -  "
        print(f"{condition:10s} {fmt('accuracy')} {fmt('f1_fake')} {fmt('auroc')} "
              f"{fmt('ece')} {fmt('uncertain_rate')} {fmt('avg_model_turns', '.1f')} "
              f"{fmt('avg_expert_calls', '.1f')}")


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    main()
