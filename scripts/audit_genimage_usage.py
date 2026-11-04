"""Audit GenImage source usage history and write a usage-aware evaluation manifest.

Read-only over all historical records.  The output is deliberately conservative:

* old split membership alone is **not** training or evaluation usage;
* a source is ``known_train`` only when a per-source training sample exists;
* SFT-data construction is recorded separately from actual training;
* calibration, development evaluation, and review are recorded as distinct roles;
* sources with none of the above are ``split_only_unknown``, not "clean".

The script writes:

  sft_data/g8_audit/genimage_usage_history_20261102.json
  docs/development/GENIMAGE_USAGE_AUDIT_20261102.md
  dataset/GenImage_q95/eval_dev_manifest_20261102.json
  dataset/GenImage_q95/eval_val_manifest_20261102.json

It never writes a test-input manifest: the contaminated test partition stays
sealed and is not used for expert selection.
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
OLD_SPLIT = ROOT / "sft_data/split_v2.json"
NEW_MANIFEST = ROOT / "dataset/GenImage_q95/manifest.json"
NEW_SPLIT = ROOT / "dataset/GenImage_q95/split_manifest.json"
AUDIT_DIR = ROOT / "sft_data/g8_audit"
OUT_JSON = AUDIT_DIR / "genimage_usage_history_20261102.json"
OUT_MD = ROOT / "docs/development/GENIMAGE_USAGE_AUDIT_20261102.md"
DEV_MANIFEST = ROOT / "dataset/GenImage_q95/eval_dev_manifest_20261102.json"
VAL_MANIFEST = ROOT / "dataset/GenImage_q95/eval_val_manifest_20261102.json"

GROUPS = {"ADM", "BigGAN", "Glide", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"}
TREATMENT_RE = re.compile(r"_(jpeg_q95|jpeg_q85|jpeg_q70|png|native)$")


# ---------------------------------------------------------------------------
# Source-ID normalisation
# ---------------------------------------------------------------------------

def norm_path(value: str) -> str:
    value = str(value or "").replace("\\", "/")
    for prefix in (str(ROOT) + "/", "/root/autodl-tmp/innovation_project/"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    return value


def strip_treatment(stem: str) -> str:
    return TREATMENT_RE.sub("", stem)


def canon_from_group_stem(group_stem: str):
    group_stem = strip_treatment(group_stem)
    if group_stem.startswith("Real/"):
        return "Real/" + group_stem[5:]
    if group_stem.startswith("Real_"):
        return "Real/" + group_stem[5:]
    if "/" in group_stem:
        group, stem = group_stem.split("/", 1)
        if group in GROUPS:
            return f"{group}/{stem}"
    for group in sorted(GROUPS, key=len, reverse=True):
        if group_stem.startswith(group + "_"):
            return f"{group}/{group_stem[len(group) + 1:]}"
    return None


def canon_from_path(path: str, source_model: str | None, aliases: dict):
    path = norm_path(path)
    match = re.search(
        r"dataset/GenImage_(?:Test|q95)/([^/]+)/([^/]+)\.(?:png|jpg|jpeg)$",
        path, re.I)
    if match:
        group, stem = match.group(1), os.path.splitext(match.group(2))[0]
        return f"{group}/{stem}" if group in GROUPS else None
    match = re.search(r"dataset/Real(?:_q95)?/([^/]+)\.(?:png|jpg|jpeg)$",
                      path, re.I)
    if match:
        return "Real/" + os.path.splitext(match.group(1))[0]
    match = re.search(r"sft_data/variants/([^/]+)\.(?:png|jpg|jpeg)$",
                      path, re.I)
    if match:
        stem = os.path.splitext(match.group(1))[0]
        found = canon_from_group_stem(stem)
        if found:
            return found
        if source_model in GROUPS or source_model == "Real":
            return f"{source_model}/{strip_treatment(stem)}"
    match = re.search(
        r"calibration/set/images(?:_d|_acceptance)?/([^/]+)\.(?:png|jpg|jpeg)$",
        path, re.I)
    if match:
        stem = os.path.splitext(match.group(1))[0]
        return aliases.get(stem) or aliases.get(strip_treatment(stem))
    basename = os.path.splitext(os.path.basename(path))[0]
    found = canon_from_group_stem(basename)
    if found:
        return found
    if source_model in GROUPS or source_model == "Real":
        return f"{source_model}/{strip_treatment(basename)}"
    return None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def read_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_json_list(path):
    payload = read_json(path)
    return payload if isinstance(payload, list) else []


def add(flags, roles, sid, flag, role):
    if not sid:
        return
    flags[sid].add(flag)
    roles[sid].add(role)


def build_aliases():
    aliases = {}
    for rel in ("calibration/set/manifest.json",
                "calibration/set/manifest_d.json",
                "calibration/set/acceptance_batch.json"):
        payload = read_json(ROOT / rel)
        for sample in payload["samples"]:
            sid = canon_from_path(sample["source_path"], sample.get("label"), aliases)
            if sid:
                aliases[sample["source_id"]] = sid
    return aliases


def add_sft_dataset(flags, roles, name, paths, excluded_names=()):
    sources = set()
    samples = 0
    for path in paths:
        if os.path.basename(path) in excluded_names:
            continue
        for row in read_json_list(path):
            samples += 1
            sid = canon_from_path(row.get("image_path", ""),
                                  row.get("source_model"), ALIASES)
            if sid:
                sources.add(sid)
                add(flags, roles, sid, "known_sft_data", f"sft_dataset:{name}")
    return sources, samples


def scan_trace_sessions(pattern, role, flags, roles, trace_counts):
    sources = set()
    for path in glob.glob(str(pattern)):
        try:
            payload = read_json(path)
        except Exception:
            continue
        sid = canon_from_path(payload.get("image_path", ""),
                              payload.get("source_model"), ALIASES)
        if not sid:
            continue
        sources.add(sid)
        trace_counts[role][sid] += 1
    for sid in sources:
        add(flags, roles, sid, "known_dev_eval", f"trace:{role}")
    return sources


def scan_smoke(flags, roles):
    sources = set()
    for path in glob.glob(str(ROOT / "sft_data/g8_smoke/**/*.json"), recursive=True):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        sid = canon_from_path(payload.get("image_path") or payload.get("image", ""),
                              payload.get("source_model"), ALIASES)
        if sid:
            sources.add(sid)
    for sid in sources:
        add(flags, roles, sid, "known_dev_eval", "g8_smoke")
    return sources


def scan_g8_trajectories(flags, roles):
    sources = set()
    for batch in ("b1", "b2"):
        directory = ROOT / "sft_data/g8/trajectories" / batch
        if not directory.exists():
            continue
        for name in os.listdir(directory):
            if not name.endswith(".json") or name == "metadata.json":
                continue
            stem = name[:-5].replace("g8_default__", "").replace("__g8_default", "")
            sid = canon_from_group_stem(stem)
            if sid:
                sources.add(sid)
    for sid in sources:
        add(flags, roles, sid, "known_dev_eval", "g8_b1_b2_trajectory")
    return sources


def scan_reviews(flags, roles):
    sources = set()
    for rel in ("sft_data/g8/review/b1/review_g8_b1.json",
                "sft_data/g8/review/b2/review_g8_b2.json"):
        path = ROOT / rel
        if not path.exists():
            continue
        decisions = read_json(path)["decisions"]
        for key in decisions:
            stem = key.replace("g8_default__", "")
            sid = canon_from_group_stem(stem)
            if sid:
                sources.add(sid)
                add(flags, roles, sid, "known_review", "g8_b1_b2_review")
    legacy = ROOT / "sft_data/review/dispositions.jsonl"
    if legacy.exists():
        for line in legacy.open(encoding="utf-8"):
            row = json.loads(line)
            stem = row.get("sample_id", "").split("__", 1)[-1]
            sid = canon_from_group_stem(stem)
            if sid:
                sources.add(sid)
                add(flags, roles, sid, "known_review", "legacy_review")
    return sources


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ALIASES = build_aliases()

old = read_json(OLD_SPLIT)
old_split = {sid: row["split"] for sid, row in old["sources"].items()}
holdout = set(old["holdout"])

new_payload = read_json(NEW_MANIFEST)
new_rows = new_payload["inputs"]
new_by_source = {row["source_id"]: row for row in new_rows}
if len(new_by_source) != len(new_rows):
    raise SystemExit("duplicate source_id in GenImage q95 manifest")
split_payload = read_json(NEW_SPLIT)

flags = collections.defaultdict(set)
roles = collections.defaultdict(set)
trace_counts = collections.defaultdict(collections.Counter)
evidence = {}

# 1. Actual training: only sources present in the G4 final_v2 samples.
train_sources = set()
train_samples = 0
final_v2 = ROOT / "archive/legacy_v1_20261015/experiments/train_final_v2"
for path in sorted(final_v2.glob("sft_*.json")):
    if path.name == "sft_invalid.json":
        continue
    for row in read_json_list(path):
        train_samples += 1
        sid = canon_from_path(row.get("image_path", ""),
                              row.get("source_model"), ALIASES)
        if sid:
            train_sources.add(sid)
            add(flags, roles, sid, "known_train", "g4_final_v2_training")
if train_samples != 237 or len(train_sources) != 20:
    raise SystemExit(f"unexpected training history: {train_samples} samples, "
                     f"{len(train_sources)} sources")
evidence["actual_training"] = {
    "dataset": "archive/legacy_v1_20261015/experiments/train_final_v2",
    "samples": train_samples,
    "sources": len(train_sources),
    "run_manifests": [
        "sft_data/lora_runs/provisional_20src/run_manifest.json",
        "sft_data/lora_runs/provisional_20src_v2/run_manifest.json",
    ],
}

# 2. SFT data constructed, not necessarily trained.
sft_sources = {}
sft_sources["train_final"], n = add_sft_dataset(
    flags, roles, "train_final",
    sorted(glob.glob(str(ROOT / "archive/legacy_v1_20261015/experiments/train_final/*.json"))),
    excluded_names={"metadata.json", "sft_rejected.json"})
sft_sources["train_final_v2"], n2 = add_sft_dataset(
    flags, roles, "train_final_v2",
    sorted(glob.glob(str(final_v2 / "*.json"))),
    excluded_names={"metadata.json", "sft_invalid.json"})
sft_sources["g7_b1"], n3 = add_sft_dataset(
    flags, roles, "g7_b1",
    sorted(glob.glob(str(ROOT / "archive/legacy_v1_20261015/experiments/train_batches_g7_b1/*.json"))),
    excluded_names={"metadata.json", "manifest.json", "selection.json",
                    "selection_v1_12sources.json", "review_only.json",
                    "sft_invalid.json"})
sft_sources["g7_b2"], n4 = add_sft_dataset(
    flags, roles, "g7_b2",
    sorted(glob.glob(str(ROOT / "archive/legacy_v1_20261015/experiments/train_batches_g7_b2/*.json"))),
    excluded_names={"metadata.json", "manifest.json", "selection.json",
                    "review_only.json", "sft_invalid.json"})
evidence["sft_data_constructed"] = {
    name: {"sources": len(values), "samples": counts}
    for (name, values), counts in zip(sft_sources.items(), (n, n2, n3, n4))
}

# 3. Calibration: all 98 holdout sources were used for G2 calibration; the
#    G7 A/B/C roles are recorded separately.  D and acceptance are eval-only.
g7 = read_json(ROOT / "calibration/set/g7_split.json")
g7_sources = {
    group: {ALIASES[alias] for alias in members if alias in ALIASES}
    for group, members in g7["groups"].items()
}
if set().union(*g7_sources.values()) != holdout:
    raise SystemExit("G7 split does not equal the 98-source holdout")
for sid in holdout:
    add(flags, roles, sid, "known_fit", "g2_calibration_holdout")
    add(flags, roles, sid, "known_calibration", "g2_calibration_holdout")
for sid in g7_sources["A"]:
    add(flags, roles, sid, "known_fit", "g7_A_fit")
for group in ("B", "C"):
    for sid in g7_sources[group]:
        add(flags, roles, sid, "known_dev_eval", f"g7_{group}_" +
            ("selection" if group == "B" else "acceptance"))

manifest_d = read_json(ROOT / "calibration/set/manifest_d.json")
d_sources = {ALIASES[s["source_id"]] for s in manifest_d["samples"]
             if s["source_id"] in ALIASES}
for sid in d_sources:
    add(flags, roles, sid, "known_calibration", "g7_D_trufor_retest")
    add(flags, roles, sid, "known_dev_eval", "g7_D_trufor_retest")
acceptance = read_json(ROOT / "calibration/set/acceptance_batch.json")
acceptance_sources = {ALIASES[s["source_id"]] for s in acceptance["samples"]
                      if s["source_id"] in ALIASES}
for sid in acceptance_sources:
    add(flags, roles, sid, "known_calibration", "g7_acceptance_batch")
    add(flags, roles, sid, "known_dev_eval", "g7_acceptance_batch")
evidence["calibration"] = {
    "holdout_sources": len(holdout),
    "g7": {group: len(values) for group, values in g7_sources.items()},
    "D_sources": len(d_sources),
    "acceptance_sources": len(acceptance_sources),
    "note": "All 98 holdout sources were used in G2 calibration; only A was used "
            "for the G7 fit. B/C are selection/acceptance, D is TruFor retest.",
}

# 4. Old val was measured wholesale by g8_eval (284 sources, 1,136 inputs).
old_val = {sid for sid, split in old_split.items() if split == "val"}
probe_val_report = ROOT / "sft_data/g8_eval/20261015_131138_probe_val/report.json"
if probe_val_report.exists():
    report = read_json(probe_val_report)
    if report.get("sources") != len(old_val):
        raise SystemExit("old-val evaluation source count does not match split_v2")
for sid in old_val:
    add(flags, roles, sid, "known_dev_eval", "old_val_g8_eval")
evidence["old_val_eval"] = {
    "sources": len(old_val),
    "report": "sft_data/g8_eval/20261015_131138_probe_val/report.json",
    "note": "The report records 284 val sources; all old-val sources are treated "
            "as development-evaluated.",
}

# 5. Traces, G8 trajectories, reviews, smoke.
trace_sources = {}
trace_sources["sft_sessions"] = scan_trace_sessions(
    ROOT / "traces/sft_sessions/*.json", "sft_sessions", flags, roles, trace_counts)
trace_sources["dry_run"] = scan_trace_sessions(
    ROOT / "traces/dry_run_sessions/*.json", "dry_run", flags, roles, trace_counts)
trace_sources["g7_acceptance"] = scan_trace_sessions(
    ROOT / "traces/g7_acceptance/*.json", "g7_acceptance", flags, roles, trace_counts)
trace_sources["g7_acceptance_before"] = scan_trace_sessions(
    ROOT / "traces/g7_acceptance_before/*.json", "g7_acceptance_before",
    flags, roles, trace_counts)
g8_traj_sources = scan_g8_trajectories(flags, roles)
review_sources = scan_reviews(flags, roles)
smoke_sources = scan_smoke(flags, roles)
evidence["agent_records"] = {
    "traces": {
        name: {"sources": len(values),
               "sessions": sum(trace_counts[role].values())}
        for name, values in trace_sources.items()
        for role in [{"sft_sessions": "sft_sessions", "dry_run": "dry_run",
                      "g7_acceptance": "g7_acceptance",
                      "g7_acceptance_before": "g7_acceptance_before"}[name]]
    },
    "g8_b1_b2_trajectory_sources": len(g8_traj_sources),
    "review_sources": len(review_sources),
    "smoke_sources": len(smoke_sources),
}

# 6. Per-source records for all 9,000 q95 inputs.
def old_label(sid):
    if sid in old_split:
        return old_split[sid]
    if sid in holdout:
        return "holdout"
    return "not_in_old"


def signature(sid):
    vals = tuple(sorted(flags.get(sid, set())))
    return vals if vals else ("split_only_unknown",)


records = []
unmatched_usage = set()
for sid, row in sorted(new_by_source.items()):
    fs = sorted(flags.get(sid, set()))
    rs = sorted(roles.get(sid, set()))
    if not fs:
        rs = ["split_only_unknown"]
    rec = {
        "source_id": sid,
        "group": row["group"],
        "label": row["label"],
        "new_split": row["split"],
        "old_split": old_label(sid),
        "source_path": row["source_path"],
        "source_sha256": row["source_sha256"],
        "source_bytes": row["source_bytes"],
        "q95_path": row["path"],
        "q95_sha256": row["sha256"],
        "q95_bytes": row["bytes"],
        "width": row["width"],
        "height": row["height"],
        "duplicate_group": row.get("duplicate_group"),
        "duplicate_of": row.get("duplicate_of"),
        "usage_flags": fs,
        "usage_roles": rs,
        "no_known_use": not fs,
    }
    records.append(rec)
for sid in set(flags):
    if sid not in new_by_source:
        unmatched_usage.add(sid)

# 7. Summaries.
def split_summary(name):
    subset = [r for r in records if r["new_split"] == name]
    signatures = collections.Counter(signature(r["source_id"]) for r in subset)
    by_group = collections.defaultdict(lambda: {"total": 0, "no_known_use": 0})
    for r in subset:
        by_group[r["group"]]["total"] += 1
        by_group[r["group"]]["no_known_use"] += int(r["no_known_use"])
    return {
        "inputs": len(subset),
        "no_known_use": sum(1 for r in subset if r["no_known_use"]),
        "known_use": sum(1 for r in subset if not r["no_known_use"]),
        "usage_signatures": {"|".join(k): v for k, v in signatures.most_common()},
        "by_group": dict(sorted(by_group.items())),
        "old_split": dict(collections.Counter(r["old_split"] for r in subset)),
    }

dev_real = sum(
    1 for r in records if r["new_split"] == "dev" and r["group"] == "Real")
summary = {
    "old_split": {
        "sources": len(old_split),
        "by_split": dict(collections.Counter(old_split.values())),
        "holdout": len(holdout),
    },
    "new_split": {name: split_summary(name)
                  for name in ("dev", "val", "test", "excluded_duplicate")},
    "actual_training": {
        "sources": len(train_sources),
        "samples": train_samples,
        "note": "Only these 20 sources have per-source samples in the G4 final_v2 "
                "training data and matching LoRA run manifests.",
    },
    "sft_data_constructed": evidence["sft_data_constructed"],
    "calibration": evidence["calibration"],
    "agent_records": evidence["agent_records"],
    "old_test_prior_use": {
        "old_test_sources": len({sid for sid, s in old_split.items() if s == "test"}),
        "with_known_use": sum(
            1 for sid, s in old_split.items()
            if s == "test" and flags.get(sid)),
        "split_only_unknown": sum(
            1 for sid, s in old_split.items()
            if s == "test" and not flags.get(sid)),
        "note": "The old test partition is not clean either; 30 sources have "
                "prior SFT-data/trace usage. It remains sealed.",
    },
    "unmatched_usage_sources": sorted(unmatched_usage),
    "shared_real": {
        "dev_inputs": dev_real,
        "dev_no_known_use": sum(
            1 for r in records
            if r["new_split"] == "dev" and r["group"] == "Real" and r["no_known_use"]),
        "val_inputs": sum(
            1 for r in records if r["new_split"] == "val" and r["group"] == "Real"),
        "val_no_known_use": sum(
            1 for r in records
            if r["new_split"] == "val" and r["group"] == "Real" and r["no_known_use"]),
        "note": "The same Real inputs are reused as negatives for every generator "
                "group; group metrics are not independent and no pooled AUROC is reported.",
    },
    "measurement_reuse": {
        "genimage_q95": {
            "existing_probe_spai_measurements": 0,
            "note": "No GenImage q95 PROBE/SPAI scores exist; both must run on dev.",
        },
        "mnw_div2k_q95": {
            "probe_scores": "sft_data/g8_external_eval/dev1_probe_20261026_2132/scores.jsonl",
            "spai_scores": "sft_data/g8_external_eval/dev1_spai_20261026_2145/scores.jsonl",
            "inputs": 100,
            "groups": 4,
            "note": "Reuse existing q95 measurements only; report separately from GenImage.",
        },
    },
    "protocol": {
        "dev_all": "new dev, 6,299 inputs",
        "dev_no_known_use": "new dev, no usage flags, primary selection set",
        "val_all": "new val, 1,350 inputs",
        "val_no_known_use": "new val, no usage flags, confirmation/selection set",
        "test_sealed": "new test stays sealed; contaminated/unknown, not an unseen test",
        "old_test_sealed": "old test stays sealed; 30 prior-use sources, not clean",
        "thresholds": "fixed registered values; no fitting on any evaluation set",
        "unseen_test_available": False,
        "unseen_test_reason": "No GenImage partition is provably unseen: new test is "
            "mostly old-train members, training records are incomplete, and old test "
            "has 30 sources with prior use. New test is retained only as a sealed "
            "contaminated development check.",
    },
}
# The walrus above is only for the dev Real count; keep the field explicit.
summary["shared_real"]["dev_inputs"] = dev_real

# 8. Hash the generated JSON payload for the manifest headers.
def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def write_manifest(path: Path, split: str):
    subset = [r for r in records if r["new_split"] == split]
    inputs = []
    for r in subset:
        inputs.append({
            "variant_id": "genimage__" + r["source_id"].replace("/", "__") + "__q95",
            "condition": "jpeg_q95",
            "group": r["group"],
            "label": r["label"],
            "source_id": r["source_id"],
            "path": r["q95_path"],
            "source_path": r["source_path"],
            "sha256": r["q95_sha256"],
            "source_sha256": r["source_sha256"],
            "bytes": r["q95_bytes"],
            "width": r["width"],
            "height": r["height"],
            "old_split": r["old_split"],
            "new_split": r["new_split"],
            "usage_flags": r["usage_flags"],
            "usage_roles": r["usage_roles"],
            "no_known_use": r["no_known_use"],
            "eval_sets": [f"{split}_all"] + (
                [f"{split}_no_known_use"] if r["no_known_use"] else []),
        })
    payload = {
        "schema": "genimage_eval_inputs/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_manifest": "dataset/GenImage_q95/manifest.json",
        "usage_audit": "sft_data/g8_audit/genimage_usage_history_20261102.json",
        "split": split,
        "selection_rule": "Use *_no_known_use for selection; *_all is descriptive.",
        "counts": {
            "inputs": len(inputs),
            "no_known_use": sum(1 for row in inputs if row["no_known_use"]),
            "by_group": dict(collections.Counter(row["group"] for row in inputs)),
        },
        "inputs": inputs,
    }
    data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
    path.write_bytes(data)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_bytes(data),
            "counts": payload["counts"]}


AUDIT_DIR.mkdir(parents=True, exist_ok=True)
manifest_info = {
    "dev": write_manifest(DEV_MANIFEST, "dev"),
    "val": write_manifest(VAL_MANIFEST, "val"),
}

summary["generated_at"] = datetime.now().isoformat(timespec="seconds")
summary["sources"] = records
summary["git_commit"] = subprocess.check_output(
    ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
summary["records"] = len(records)
summary["manifests"] = manifest_info
summary["files"] = {
    "json": str(OUT_JSON.relative_to(ROOT)),
    "markdown": str(OUT_MD.relative_to(ROOT)),
    "dev_manifest": str(DEV_MANIFEST.relative_to(ROOT)),
    "val_manifest": str(VAL_MANIFEST.relative_to(ROOT)),
}

OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")

# 9. Markdown report.
s = summary
lines = [
    "# GenImage 使用历史审计（2026-11-02）",
    "",
    "本文件由 `scripts/audit_genimage_usage.py` 生成；旧划分、轨迹、校准、审核记录只读。",
    "逐源记录和双哈希见 `sft_data/g8_audit/genimage_usage_history_20261102.json`。",
    "",
    "## 结论",
    "",
    "- 可证明的实际训练只覆盖 **%d 个源**（G4 final_v2 的 %d 个样本）；"
    "旧 train 成员本身不记为训练。" % (s["actual_training"]["sources"],
                                      s["actual_training"]["samples"]),
    "- 新 test 的旧划分来源：train **%d**、val **%d**、test **%d**、holdout **%d**。"
    % (s["new_split"]["test"]["old_split"].get("train", 0),
       s["new_split"]["test"]["old_split"].get("val", 0),
       s["new_split"]["test"]["old_split"].get("test", 0),
       s["new_split"]["test"]["old_split"].get("holdout", 0)),
    "- 新 test 中 **%d** 个源有已知使用，**%d** 个仅有划分归属、使用未知。"
    % (s["new_split"]["test"]["known_use"], s["new_split"]["test"]["no_known_use"]),
    "- **没有可靠、可证明未见的 GenImage q95 最终测试**；新 test 只保留为隔离的"
    "污染/未知开发检查，不用于专家选择。",
    "- 旧 test 286 源中 **%d** 个有先验使用，**%d** 个仅划分归属；旧 test 也不是"
    "天然干净，继续封存。" % (s["old_test_prior_use"]["with_known_use"],
                              s["old_test_prior_use"]["split_only_unknown"]),
    "",
    "## 新划分 × 旧划分",
    "",
    "| 新划分 | 输入 | 旧 train | 旧 val | 旧 test | holdout | 无已知使用 | 有已知使用 |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for name in ("dev", "val", "test", "excluded_duplicate"):
    row = s["new_split"][name]
    lines.append(
        f"| {name} | {row['inputs']} | {row['old_split'].get('train', 0)} | "
        f"{row['old_split'].get('val', 0)} | {row['old_split'].get('test', 0)} | "
        f"{row['old_split'].get('holdout', 0)} | {row['no_known_use']} | "
        f"{row['known_use']} |")
lines += [
    "",
    "## 使用类别",
    "",
    "| 类别 | 说明 |",
    "| --- | --- |",
    "| `known_train` | 有逐源训练样本和 LoRA run manifest；仅 20 源 |",
    "| `known_sft_data` | 进入过构造的 SFT 数据，但无实际训练证据；不等于已训练 |",
    "| `known_fit` | 98 个 G2 校准 holdout；其中 A 组用于 G7 拟合，B/C 用于选择/验收 |",
    "| `known_dev_eval` | 旧 val 284 源、轨迹/Agent session、G7 验收/D、smoke 等 |",
    "| `known_review` | b1/b2 人工审核与旧 review dispositions |",
    "| `split_only_unknown` | 只有旧/新划分归属，无已知使用；**不等于 clean/unseen** |",
    "",
    "## 评测口径",
    "",
    "| 集合 | 用途 |",
    "| --- | --- |",
    f"| `dev_all`（{s['new_split']['dev']['inputs']}） | 描述性全量对照 |",
    f"| `dev_no_known_use`（{s['new_split']['dev']['no_known_use']}） | 主专家选择集 |",
    f"| `val_all`（{s['new_split']['val']['inputs']}） | 描述性/确认 |",
    f"| `val_no_known_use`（{s['new_split']['val']['no_known_use']}） | 确认/选择集 |",
    f"| `test_sealed`（{s['new_split']['test']['inputs']}） | 隔离；不是未见测试，不用 |",
    "| `old_test_sealed`（286） | 继续封存；30 源有先验使用 |",
    "| `mnw_q95_existing`（100） | 单独报告；不混入 GenImage、不训练/拟合 |",
    "",
    "共享 Real：dev %d 张、其中无已知使用 %d 张；val %d 张、其中无已知使用 %d 张。"
    "同一批 Real 在八个生成器组间复用，组间不独立；不报告跨组合并 AUROC。"
    % (s["shared_real"]["dev_inputs"], s["shared_real"]["dev_no_known_use"],
       s["shared_real"]["val_inputs"], s["shared_real"]["val_no_known_use"]),
    "",
    "## 证据与限制",
    "",
    "- 旧 `split_v2`：%d 源（train %d / val %d / test %d），holdout %d。"
    % (s["old_split"]["sources"], s["old_split"]["by_split"].get("train", 0),
       s["old_split"]["by_split"].get("val", 0),
       s["old_split"]["by_split"].get("test", 0), s["old_split"]["holdout"]),
    "- GenImage q95 没有既有 PROBE/SPAI 测量；MNW/DIV2K 的 q95 既有测量只用于"
    "单独报告，不重跑、不混入。",
    "- 旧 val 284 源全部进入过 `sft_data/g8_eval` 的 val 评测；新 test 中的 37 个"
    "old-val 源因此有开发评测历史。",
    "- 训练记录不完整：除 20 个 G4 源外，不能证明其余旧 train 源未用于训练；"
    "因此 `split_only_unknown` 不是 clean。",
    "- 旧 test 的 30 个先验使用源也意味着旧 test 不能默认当作未见最终测试。",
    "",
    "## 产物",
    "",
    f"- JSON：`{s['files']['json']}`",
    f"- Dev manifest：`{s['files']['dev_manifest']}`（{manifest_info['dev']['counts']['inputs']} 输入）",
    f"- Val manifest：`{s['files']['val_manifest']}`（{manifest_info['val']['counts']['inputs']} 输入）",
    "- 不生成 test manifest；test 保持隔离。",
]
OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps({
    "dev": manifest_info["dev"]["counts"],
    "val": manifest_info["val"]["counts"],
    "test": s["new_split"]["test"],
    "actual_training_sources": s["actual_training"]["sources"],
    "unseen_test_available": s["protocol"]["unseen_test_available"],
}, ensure_ascii=False, indent=2))
