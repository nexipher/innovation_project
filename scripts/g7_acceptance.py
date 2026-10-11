#!/usr/bin/env python3
"""
G7 entry 3, step 3: the small-batch acceptance run.

Two arms on the same 120 samples (24 sources x 5 formats):

  agent   the full protocol — PROBE available as the main detector, the three
          statistical experts offered as corroboration, verdict from the
          halting policy
  probe   PROBE alone, read through the frozen A-fitted mapping and the
          B-selected threshold — entry 2's reference, re-measured on this batch

What the run is for: not detection quality (that is entry 6's question) but
whether the wiring holds up on real sessions — calls that land as tokens,
verdicts that can stand with or without the corroborating experts, abstentions
that are honest, stop reasons that can be explained from the trace.

The batch: all 18 development (B) sources plus 6 fresh sources from
`split_v2`'s val partition, three Real and three Fake, every one of them
disjoint from D and from the test partition.  Formats are the calibration
set's own five encodings, so a refusal on one of them means what it meant
everywhere else.

Usage:
  python scripts/g7_acceptance.py --dry-run            # CPU plumbing (mock)
  python scripts/g7_acceptance.py                      # GPU (authorize first)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT
from mllm.message_builder import build_forensic_prompt
from state_machine.controller import ForensicStateMachine
from utils import config_fingerprint
from utils.logger import SessionLogger

SET_DIR = os.path.join(PROJECT_ROOT, "calibration", "set")
MANIFEST_PATH = os.path.join(SET_DIR, "manifest.json")
SPLIT_PATH = os.path.join(SET_DIR, "g7_split.json")
SPLIT_V2_PATH = os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json")
MANIFEST_D_PATH = os.path.join(SET_DIR, "manifest_d.json")
BATCH_PATH = os.path.join(SET_DIR, "acceptance_batch.json")
IMAGES_DIR = os.path.join(SET_DIR, "images_acceptance")
REPORT_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_acceptance_report.json")
DRY_RUN_REPORT_PATH = os.path.join(
    PROJECT_ROOT, "calibration", "g7_acceptance_report_dry_run.json")
DRY_RUN_SESSIONS_DIR = os.path.join(PROJECT_ROOT, "traces", "dry_run_sessions")
FREEZE_PATH = os.path.join(PROJECT_ROOT, "calibration", "g7_frozen_scheme.json")

TREATMENTS = ("native", "png", "jpeg_q95", "jpeg_q85", "jpeg_q70")
VAL_TOP_UP = 6
DEFAULT_SEED = 20261011


# ---------------------------------------------------------------------------
# The batch
# ---------------------------------------------------------------------------

def select_sources(seed: int = DEFAULT_SEED) -> List[dict]:
    """
    All of B, plus six fresh val sources — checked, not assumed, disjoint.

    B is the development group: it chose the operating threshold, so a verdict
    there is a re-measurement rather than a discovery.  The val sources are
    here to keep the batch at the 20-40 the plan asks for; they were never
    part of any fit, choice or earlier evaluation, and D's 64 sources are
    excluded so the TruFor re-check stays its own experiment.
    """
    with open(MANIFEST_PATH, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(SPLIT_PATH, encoding="utf-8") as handle:
        split = json.load(handle)
    with open(SPLIT_V2_PATH, encoding="utf-8") as handle:
        split_v2 = json.load(handle)

    b_sources = set(split["groups"]["B"])
    by_source: Dict[str, dict] = {}
    for sample in manifest["samples"]:
        if sample["source_id"] in b_sources:
            by_source.setdefault(sample["source_id"], sample)

    used_paths = {sample["source_path"] for sample in by_source.values()}
    if os.path.exists(MANIFEST_D_PATH):
        with open(MANIFEST_D_PATH, encoding="utf-8") as handle:
            used_paths |= {s["source_path"] for s in json.load(handle)["samples"]}

    sources = [{"source_id": source_id, "label": entry["label"],
                "generator": entry["generator"], "source_path": entry["source_path"],
                "resolution": entry["resolution"], "origin": "B"}
               for source_id, entry in sorted(by_source.items())]

    val = {key: entry for key, entry in split_v2["sources"].items()
           if entry["split"] == "val" and entry["path"] not in used_paths}
    rng = np.random.default_rng(seed)
    real_pool = sorted((k, e) for k, e in val.items() if e["label"] == "Real")
    picked = sorted(rng.choice(len(real_pool), size=3, replace=False).tolist())
    for index, position in enumerate(picked):
        key, entry = real_pool[position]
        sources.append({"source_id": f"acc_real_{index:04d}", "label": "Real",
                        "generator": "Real", "source_path": entry["path"],
                        "resolution": entry["resolution"], "origin": "val"})

    generators = sorted({e["generator"] for e in val.values() if e["label"] == "Fake"})
    for generator in generators[:3]:
        pool = sorted((k, e) for k, e in val.items()
                      if e["label"] == "Fake" and e["generator"] == generator)
        key, entry = pool[int(rng.integers(len(pool)))]
        sources.append({"source_id": f"acc_fake_{generator.lower()}",
                        "label": "Fake", "generator": generator,
                        "source_path": entry["path"], "resolution": entry["resolution"],
                        "origin": "val"})
    return sources


def check_sources(sources: List[dict]) -> List[str]:
    """Disjointness and coverage, asserted rather than described."""
    problems = []
    origins = Counter(source["origin"] for source in sources)
    if origins["B"] != 18 or origins["val"] != VAL_TOP_UP:
        problems.append(f"批次成分不对：{dict(origins)}")
    if len({s["source_path"] for s in sources}) != len(sources):
        problems.append("批次内有重复源图")

    with open(SPLIT_V2_PATH, encoding="utf-8") as handle:
        split_v2 = json.load(handle)
    test_paths = {e["path"] for e in split_v2["sources"].values() if e["split"] == "test"}
    train_paths = {e["path"] for e in split_v2["sources"].values() if e["split"] == "train"}
    for source in sources:
        path = source["source_path"]
        if path in test_paths:
            problems.append(f"{source['source_id']} 落在最终 test 分区")
        if path in train_paths:
            problems.append(f"{source['source_id']} 落在训练分区")
    return problems


def build_batch(sources: List[dict], dry_run: bool = False) -> List[dict]:
    """Five format encodings per source, the calibration set's own code path."""
    from scripts.build_calibration_set import build_cells, resolution_bucket
    from utils.image_utils import ImageUtils

    os.makedirs(IMAGES_DIR, exist_ok=True)
    samples: List[dict] = []
    for source in sources:
        image = ImageUtils.load_image(source["source_path"])
        if image is None:
            raise SystemExit(f"读不到源图：{source['source_path']}")
        height, width = image.shape[:2]
        sample_id = source["source_id"]
        common = {
            "source_id": sample_id, "source_path": source["source_path"],
            "label": source["label"], "generator": source["generator"],
            "resolution": source["resolution"],
            "resolution_bucket": resolution_bucket(height, width),
            "origin": source["origin"],
        }
        if not dry_run:
            import cv2
            extension = os.path.splitext(source["source_path"])[1].lower()
            native_path = os.path.join(IMAGES_DIR, f"{sample_id}_native{extension}")
            if extension in (".jpg", ".jpeg"):
                ok = bool(cv2.imwrite(native_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95]))
                container = "jpg"
            else:
                ok = bool(cv2.imwrite(native_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 6]))
                container = "png"
            if ok:
                samples.append({**common, "sample_id": f"{sample_id}_native",
                                "cell": f"{source['label'].lower()}_native",
                                "treatment": "native", "container": container,
                                "quality": None,
                                "path": os.path.relpath(native_path, PROJECT_ROOT).replace(os.sep, "/")})
            for cell in build_cells(image, sample_id, out_dir=IMAGES_DIR):
                samples.append({**common, "sample_id": f"{sample_id}_{cell['treatment']}",
                                "cell": f"{source['label'].lower()}_{cell['treatment']}",
                                "treatment": cell["treatment"], "container": cell["container"],
                                "quality": cell["quality"], "path": cell["path"]})
        else:
            for treatment in TREATMENTS:
                samples.append({**common, "sample_id": f"{sample_id}_{treatment}",
                                "cell": f"{source['label'].lower()}_{treatment}",
                                "treatment": treatment, "container": "png", "quality": None,
                                "path": source["source_path"]})
    return samples


def load_batch(samples: List[dict]) -> None:
    """Write the batch manifest, so the GPU run and the report agree on it."""
    if os.path.exists(BATCH_PATH):
        with open(BATCH_PATH, encoding="utf-8") as handle:
            previous = json.load(handle)
        if previous.get("samples") == samples:
            return
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "seed": DEFAULT_SEED,
        "note": ("G7 入口 3 小批验收批次：B 组 18 源（全部）+ val 分区 6 源（3 Real + 3 Fake，"
                 "与 D 组和最终 test 分区互斥），每源 5 个格式，共 120 样本。"),
        "counts": {"sources": len({s['source_id'] for s in samples}),
                   "samples": len(samples),
                   "by_label": dict(Counter(s["label"] for s in samples)),
                   "by_treatment": dict(Counter(s["treatment"] for s in samples))},
        "samples": samples,
    }
    with open(BATCH_PATH, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)


# ---------------------------------------------------------------------------
# The reference arm
# ---------------------------------------------------------------------------

def frozen_probe_mapping() -> dict:
    """
    PROBE's standalone reading: the A-fitted logistic mapping, frozen in
    `g7_frozen_scheme.json`, with the threshold B selected (0.251).

    Note this is *not* the same mapping the tokens carry: the token pipeline
    uses the A-fitted quantile bins, exactly as the three statistical experts
    do.  Two A-fitted readings of the same score, reported side by side and
    never averaged together.
    """
    with open(FREEZE_PATH, encoding="utf-8") as handle:
        freeze = json.load(handle)
    entry = freeze["conditions"]["probe"]
    return {
        "features": entry["features"],
        "scaler_mean": np.array(entry["scaler_mean"], dtype=float),
        "scaler_scale": np.array(entry["scaler_scale"], dtype=float),
        "coefficients": np.array([entry["coefficients"][f] for f in entry["features"]], dtype=float),
        "intercept": float(entry["intercept"]),
        "threshold": float(entry["threshold_b"]),
        "frozen_at": freeze.get("generated_at"),
        "fit_group": freeze.get("fitted_on"),
    }


def probe_probability(mapping: dict, score: float) -> float:
    z = (np.array([score]) - mapping["scaler_mean"]) / mapping["scaler_scale"]
    logit = float(np.dot(mapping["coefficients"], z)) + float(mapping["intercept"])
    return float(1.0 / (1.0 + np.exp(-logit)))


def probe_reference(expert, samples: List[dict], mapping: dict,
                    table=None, progress: bool = True) -> Dict[str, dict]:
    """
    PROBE's raw score alone, read two ways.

    The agent's tokens carry the A-fitted **quantile bins** — the mapping every
    expert in the toolkit uses — while entry 2's frozen reference is a logistic
    mapping at the B-selected threshold.  Both are fitted on A, and they do not
    agree sample by sample, so a disagreement between the agent and the
    reference could come from the mapping rather than from anything Qwen did.
    Reading the same cached score through the *same bins the tokens use* (via
    the runtime table, not a copy of it) separates the two.
    """
    import cv2

    reference: Dict[str, dict] = {}
    for index, sample in enumerate(samples, 1):
        image = cv2.imread(os.path.join(PROJECT_ROOT, sample["path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        started = time.perf_counter()
        result = expert.analyze(image)
        elapsed = time.perf_counter() - started
        score = float(result.raw_metric)
        probability = probe_probability(mapping, score)

        entry = {
            "raw_score": round(score, 6),
            # Entry 2's frozen reading: logistic mapping, threshold 0.251.
            "p_fake_logistic": round(probability, 4),
            "verdict_logistic": "Fake" if probability >= mapping["threshold"] else "Real",
            "elapsed_s": round(elapsed, 3),
        }
        if table is not None:
            lookup = table.lookup("probe_dino_expert", score)
            bins_probability = (lookup or {}).get("calibrated_likelihood", {}).get("Fake")
            entry["p_fake_bins"] = None if bins_probability is None else round(bins_probability, 4)
            entry["verdict_bins"] = None if bins_probability is None else (
                "Fake" if bins_probability >= 0.5 else "Real")
        reference[sample["sample_id"]] = entry
        if progress and index % 40 == 0:
            print(f"    参考臂 [{index}/{len(samples)}]")
    return reference


def mapping_for_report(mapping: dict) -> dict:
    """The same mapping without the numpy arrays — the report is a JSON file."""
    return {
        "features": list(mapping["features"]),
        "scaler_mean": [float(v) for v in mapping["scaler_mean"]],
        "scaler_scale": [float(v) for v in mapping["scaler_scale"]],
        "coefficients": [float(v) for v in mapping["coefficients"]],
        "intercept": float(mapping["intercept"]),
        "threshold": float(mapping["threshold"]),
        "frozen_at": mapping.get("frozen_at"),
        "fit_group": mapping.get("fit_group"),
        "note": ("入口 2 冻结的 A 组拟合映射；token 走的是同一拟合组的分位分箱，"
                 "两个读数并列报告，不合并"),
    }


# ---------------------------------------------------------------------------
# The agent arm
# ---------------------------------------------------------------------------

def build_experts(dry_run: bool, probe_score: Optional[float] = None) -> Dict[str, Any]:
    from experts.frequency_v2 import FrequencyExpertV2
    from experts.jpeg import JPEGExpert
    from experts.noise import NoiseExpert

    experts: Dict[str, Any] = {
        "frequency_expert_v2": FrequencyExpertV2(),
        "noise_expert": NoiseExpert(),
        "jpeg_expert": JPEGExpert(),
    }
    if dry_run:
        from experts.probe_dino import MockProbeExpert
        experts["probe_dino_expert"] = MockProbeExpert(score=probe_score)
    else:
        from experts.probe_dino import ProbeDinoExpert
        # On the card, with Qwen: measured 0.02-0.09 s/image warm and 1.6 GB
        # resident, which is what makes it cheap enough to be the first call
        # of every session.  (It defaults to CPU, which turned the reference
        # arm into a minutes-per-batch job the first time this ran.)
        probe = ProbeDinoExpert(device="cuda")
        if probe.unavailable_reason():
            raise SystemExit(f"PROBE 不可用：{probe.unavailable_reason()}")
        experts["probe_dino_expert"] = probe
    return experts


def run_agent(samples: List[dict], experts: Dict[str, Any],
              client_factory: Callable[[], Any],
              dry_run: bool, sft_dir: Optional[str],
              on_record: Optional[Callable[[dict], None]] = None,
              progress: bool = True) -> List[dict]:
    """
    One session per sample, with PROBE served and the clues optional.

    `experts` is passed in rather than built here: PROBE's backbone is a
    1.6 GB load, and the reference arm already built one — a second instance
    would pay for the same weights twice.
    """
    client = client_factory()
    records: List[dict] = []

    for index, sample in enumerate(samples, 1):
        image_path = os.path.join(PROJECT_ROOT, sample["path"])
        fsm = ForensicStateMachine(
            client, experts,
            logger=SessionLogger(sft_dir=sft_dir) if sft_dir else None)
        started = time.perf_counter()
        result = fsm.run(image_path, ground_truth=sample["label"])
        elapsed = time.perf_counter() - started

        verdict = result["final_verdict"]
        tokens = result["evidence_chain"]
        record = {
            "sample_id": sample["sample_id"],
            "source_id": sample["source_id"],
            "origin": sample.get("origin"),
            "generator": sample["generator"],
            "label": sample["label"],
            "treatment": sample["treatment"],
            "verdict": verdict.get("verdict"),
            "confidence": verdict.get("confidence"),
            "posterior": verdict.get("posterior"),
            "halting_reason": result.get("halting_reason"),
            "policy_reasons": verdict.get("policy_reasons"),
            "calls": [token["source"] for token in tokens],
            "call_tags": [ForensicStateMachine.EXPERT_SOURCE_MAP.get(token["source"], token["source"])
                          for token in tokens],
            "expert_calls": result.get("expert_call_count", 0),
            "suppressed_duplicates": result.get("suppressed_duplicate_count", 0),
            "skipped_duplicates": result.get("skipped_duplicate_calls", 0),
            "model_turns": result.get("model_turn_count", 0),
            "elapsed_s": round(elapsed, 2),
            "probe_token_p_fake": next(
                (token["calibrated_likelihood"]["Fake"] for token in tokens
                 if token["source"] == "probe_dino_expert"
                 and token.get("calibrated_likelihood")), None),
            "sft_data_path": result.get("sft_data_path"),
        }
        records.append(record)
        if on_record is not None:
            on_record(record)
        if progress and index % 20 == 0:
            print(f"    agent 臂 [{index}/{len(samples)}]")
    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(records: List[dict], reference: Dict[str, dict]) -> dict:
    agreement = Counter()
    agreement_bins = Counter()
    mapping_only = Counter()
    per_treatment: Dict[str, Counter] = defaultdict(Counter)
    halting = Counter()
    call_counts = Counter()
    first_call = Counter()

    for record in records:
        reference_entry = reference.get(record["sample_id"], {})
        probe_verdict = reference_entry.get("verdict_logistic")
        probe_verdict_bins = reference_entry.get("verdict_bins")
        agent_verdict = record["verdict"]

        # The headline comparison is agent vs the *same-mapping* reading: it is
        # the one that isolates scheduling and interpretation from calibration.
        if agent_verdict == "Uncertain":
            agreement_bins["agent_abstained"] += 1
            agreement["agent_abstained"] += 1
        elif probe_verdict_bins is None:
            agreement_bins["no_reference"] += 1
        elif agent_verdict == probe_verdict_bins:
            agreement_bins["agree"] += 1
        else:
            agreement_bins["disagree"] += 1

        if agent_verdict != "Uncertain" and probe_verdict is not None:
            if agent_verdict == probe_verdict:
                agreement["agree"] += 1
            else:
                agreement["disagree"] += 1

        if probe_verdict is not None and probe_verdict_bins is not None:
            mapping_only["same" if probe_verdict == probe_verdict_bins else "different"] += 1

        per_treatment[record["treatment"]][f"agent_{agent_verdict}"] += 1
        halting[record["halting_reason"]] += 1
        for tag in record["call_tags"] or ["<none>"]:
            call_counts[tag] += 1
        if record["call_tags"]:
            first_call[record["call_tags"][0]] += 1

    # A call the model emitted that produced no token is the silent-drop
    # failure this wiring is most likely to hide; expert_call_count counts
    # executed calls, so the two numbers have to be compared explicitly.
    landed = Counter(tag for record in records for tag in (record["call_tags"] or []))
    probe_sessions = sum(1 for record in records if "probe_dino_expert" in (record["calls"] or []))
    return {
        "samples": len(records),
        "probe_called_in": probe_sessions,
        "probe_token_rate": round(probe_sessions / len(records), 4) if records else None,
        "agreement_with_probe_bins_mapping": dict(agreement_bins),
        "agreement_with_probe_logistic": dict(agreement),
        "the_two_references_agree": dict(mapping_only),
        "agent_verdicts": dict(Counter(record["verdict"] for record in records)),
        "halting_reasons": dict(halting),
        "call_counts": dict(call_counts),
        "first_calls": dict(first_call),
        "landed_tokens_per_tag": dict(landed),
        "per_treatment": {t: dict(counter) for t, counter in sorted(per_treatment.items())},
        "skipped_duplicate_calls": sum(r.get("skipped_duplicates", 0) for r in records),
        "executed_calls": sum(r.get("expert_calls", 0) for r in records),
        "mean_seconds_per_sample": round(
            float(np.mean([r["elapsed_s"] for r in records])), 2) if records else None,
        "mean_model_turns": round(
            float(np.mean([r["model_turns"] for r in records])), 2) if records else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="MockMLLMClient + MockProbeExpert (CPU plumbing only)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 个样本（冒烟用；报告里会写明）")
    parser.add_argument("--build-only", action="store_true",
                        help="只选源、生成图像、写批次清单（CPU），不跑会话")
    args = parser.parse_args()

    sources = select_sources(args.seed)
    problems = check_sources(sources)
    if problems:
        raise SystemExit("批次不合法：" + "；".join(problems))
    samples = build_batch(sources, dry_run=args.dry_run)
    if args.limit:
        samples = samples[:args.limit]
    if not args.dry_run:
        load_batch(samples)
    print(f"批次：{len(sources)} 源（B 组 {sum(1 for s in sources if s['origin'] == 'B')} + "
          f"val {sum(1 for s in sources if s['origin'] == 'val')}）｜样本 {len(samples)}"
          f"｜标签 {dict(Counter(s['label'] for s in samples))}")
    if args.build_only:
        print(f"批次清单：{os.path.relpath(BATCH_PATH, PROJECT_ROOT)}"
              f"（图像在 {os.path.relpath(IMAGES_DIR, PROJECT_ROOT)}/，未跑会话）")
        return

    mapping = frozen_probe_mapping()
    from utils.reliability import ReliabilityTable
    table = ReliabilityTable.load()          # the same table the tokens read
    experts = build_experts(args.dry_run)    # built once: PROBE's load is shared
    reference_expert = experts["probe_dino_expert"]
    print(f"参考臂：PROBE 单独，两种读数 —— 冻结 LR 映射（阈值 {mapping['threshold']}）"
          f"与 token 同源的 A 组分位分箱")
    reference = probe_reference(reference_expert, samples, mapping, table=table,
                                progress=not args.dry_run)

    if args.dry_run:
        from mllm.mock_client import MockMLLMClient

        def client_factory():
            return MockMLLMClient(mode="probe_then_corroborate", seed=args.seed)
        mode = "dry_run"
        sft_dir = DRY_RUN_SESSIONS_DIR
    else:
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("CUDA 不可用。真实验收需要 GPU（agent.md §3.2：先取得授权）。")
        from mllm.qwen_client import QwenVLClient
        tags = [ForensicStateMachine.EXPERT_SOURCE_MAP[name] for name in experts]
        prompt = build_forensic_prompt(tags)

        def client_factory():
            return QwenVLClient(system_prompt=prompt)
        mode = "gpu"
        sft_dir = os.path.join(PROJECT_ROOT, "traces", "g7_acceptance")
    print(f"agent 臂：{'mock' if args.dry_run else 'Qwen2.5-VL-7B'} "
          f"（PROBE 主检测器 + 三专家辅助线索）")

    os.makedirs(sft_dir, exist_ok=True)
    records = run_agent(samples, experts, client_factory, args.dry_run, sft_dir)

    fingerprint = config_fingerprint.compute(experts, None)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mode": mode,
        "limited_to": args.limit,
        "seed": args.seed,
        "sources": len(sources),
        "samples": len(samples),
        "seeded_with": {"sources": [s["source_id"] for s in sources]},
        "probe_reference": mapping_for_report(mapping),
        "config_fingerprint": fingerprint,
        "summary": summarise(records, reference),
        "records": records,
        "reference": reference,
    }
    output = args.output or (DRY_RUN_REPORT_PATH if args.dry_run else REPORT_PATH)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=1, allow_nan=False)

    summary = report["summary"]
    print(f"\nPROBE 被调用：{summary['probe_called_in']}/{summary['samples']}  "
          f"｜与同映射的 PROBE 单独：{summary['agreement_with_probe_bins_mapping']}")
    print(f"与冻结 LR 读数：{summary['agreement_with_probe_logistic']}  "
          f"｜两个参考臂之间：{summary['the_two_references_agree']}")
    print(f"判定分布：{summary['agent_verdicts']}  停止原因：{summary['halting_reasons']}")
    print(f"调用计数：{summary['call_counts']}  首次调用：{summary['first_calls']}")
    print(f"平均耗时 {summary['mean_seconds_per_sample']} s/样本，"
          f"{summary['mean_model_turns']} 轮")
    print(f"\nReport: {os.path.relpath(output, PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
