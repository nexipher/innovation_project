"""
Batch identity for the review pipeline (G7 entry 4).

A review decision is only meaningful about *specific content*.  Until now a
decision was keyed by sample id alone, so regenerating a sample under the same
id — which is the normal case, the id names the variant and the policy — would
silently inherit the old verdict.  Two things are needed to stop that:

  content_sha256(sample)   a digest of what would actually be trained
  batch_fingerprint(samples)  the generation version: every sample's digest,
                              in id order, hashed once

A batch is a directory plus that fingerprint.  The historical 237 samples stay
where they are (`sft_data/train/final_v2/`) and are addressed as the legacy
batch; new work gets its own directory, so nothing is overwritten and no old
decision is inherited by accident.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional

from config import PROJECT_ROOT

# The reviewed-by-hand set from G4 (237 samples).  Kept in place, addressed by
# name, never regenerated in-place.
LEGACY_BATCH = "g4_trial"

BUCKET_FILES = ("sft_tool_positive.json", "sft_no_tool_positive.json",
                "sft_honest_abstention.json")

# Fields that are about the *record*, not about the content being trained:
# the review block is what a decision adds, and the bucket filename says where
# the sample was filed.  Everything else participates in the digest, because
# everything else can end up in the rendered conversation.
VOLATILE_FIELDS = ("review", "_bucket_file", "review_source")


def content_sha256(sample: dict) -> str:
    """Digest of everything that could reach training for this sample."""
    payload = {key: value for key, value in sample.items()
               if key not in VOLATILE_FIELDS}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def batch_fingerprint(samples: Iterable[dict]) -> str:
    """The generation version: content digests of the whole batch, in id order."""
    rows = sorted(f"{sample.get('id')}:{content_sha256(sample)}" for sample in samples)
    digest = hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()
    return digest[:16]


def is_trajectory_record(record) -> bool:
    """
    A session, not a summary that happens to sit in the same directory.

    The generator now writes its report beside the trajectories, and three
    loaders read that directory by filename alone — each of them parsed the
    report as a session and crashed.  The shape is the filter.
    """
    return isinstance(record, dict) and "trajectory_id" in record


def load_trajectory_records(directory: str) -> Dict[str, dict]:
    """trajectory_id -> record, for every real session in the directory.

    Subdirectories are not descended into: an archived arm keeps its own
    directory so a regenerated arm can live beside what it replaced.
    """
    records: Dict[str, dict] = {}
    if not os.path.isdir(directory):
        return records
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(directory, name), encoding="utf-8") as handle:
                record = json.load(handle)
        except (OSError, ValueError):
            continue
        if is_trajectory_record(record):
            records[record["trajectory_id"]] = record
    return records


def batch_paths(batch_id: str = LEGACY_BATCH,
                generation: Optional[str] = None) -> Dict[str, str]:
    """
    Where a batch's samples, reviews and output live.

    Two layouts, never mixed (§ 阶段 1，2026-10-15):

      * `generation` given — the nested layout, `sft_data/<generation>/...`,
        with a **plain** batch name (`b1`, not `g8_b1`).  The generation is a
        directory and an explicit argument; a batch name carries one meaning.
      * `generation` omitted — the historical flat layout
        (`sft_data/trajectories_<batch_id>` …), which exists so the frozen
        g7-era records keep resolving.  Writing scripts require an explicit
        `--generation`, so nothing new can be created in the flat layout by
        accident.

    The legacy batch keeps its historical paths either way (it has no
    generation of its own beyond `g7`).
    """
    if generation is not None:
        root = os.path.join(PROJECT_ROOT, "sft_data", generation)
        return {
            "samples": os.path.join(root, "train", "batches", batch_id),
            "review": os.path.join(root, "review", batch_id),
            "reviewed": os.path.join(root, "train", "batches", batch_id + "_reviewed"),
            "trajectories": os.path.join(root, "trajectories", batch_id),
            "variants_dir": os.path.join(root, "variants", batch_id),
            "variants": os.path.join(root, "variants", batch_id, "manifest.json"),
            "admission": os.path.join(root, f"admission_report_{batch_id}.json"),
        }
    if batch_id == LEGACY_BATCH:
        return {
            "samples": os.path.join(PROJECT_ROOT, "sft_data", "train", "final_v2"),
            "review": os.path.join(PROJECT_ROOT, "sft_data", "review"),
            "reviewed": os.path.join(PROJECT_ROOT, "sft_data", "train",
                                     "final_v2_reviewed"),
            "trajectories": os.path.join(PROJECT_ROOT, "sft_data", "trajectories"),
            "variants_dir": os.path.join(PROJECT_ROOT, "sft_data", "variants"),
            "variants": os.path.join(PROJECT_ROOT, "sft_data", "variants",
                                     "manifest.json"),
            "admission": os.path.join(PROJECT_ROOT, "sft_data", "admission_report.json"),
        }
    # Historical flat layout: every stage of a batch lives under its own name.
    # Variants, trajectories and the admission report are what the samples are
    # built *from*, so sharing them would leave a batch depending on another
    # batch's files — and overwriting them would rewrite the 237's inputs.
    root = os.path.join(PROJECT_ROOT, "sft_data", "train", "batches", batch_id)
    return {
        "samples": root,
        "review": os.path.join(PROJECT_ROOT, "sft_data", "review", batch_id),
        "reviewed": root + "_reviewed",
        "trajectories": os.path.join(PROJECT_ROOT, "sft_data",
                                     f"trajectories_{batch_id}"),
        "variants_dir": os.path.join(PROJECT_ROOT, "sft_data",
                                     f"variants_{batch_id}"),
        "variants": os.path.join(PROJECT_ROOT, "sft_data",
                                 f"variants_{batch_id}", "manifest.json"),
        "admission": os.path.join(PROJECT_ROOT, "sft_data",
                                  f"admission_report_{batch_id}.json"),
    }


def isolated_paths(batch_id: str) -> List[str]:
    """
    Every path a batch writes to — used to prove two batches cannot collide.

    The isolation claim is only worth making if it is checkable, and the place
    to check it is here rather than in a paragraph of the plan.
    """
    return list(batch_paths(batch_id).values())


def load_samples(batch_id: str = LEGACY_BATCH,
                 samples_dir: Optional[str] = None) -> List[dict]:
    """Every bucket file of a batch, with the bucket name attached."""
    directory = samples_dir or batch_paths(batch_id)["samples"]
    samples: List[dict] = []
    for name in BUCKET_FILES:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for sample in json.load(handle):
                samples.append({**sample, "_bucket_file": name})
    return samples


def write_manifest(batch_id: str, directory: str, *, samples: List[dict],
                   generation: Optional[dict] = None) -> dict:
    """
    The batch's own record: what it contains, and the version it is.

    Written next to the samples so a later reader can tell whether the set on
    disk is the one the reviews were recorded against, without recomputing
    anything by hand.
    """
    manifest = {
        "batch_id": batch_id,
        "fingerprint": batch_fingerprint(samples),
        "samples": len(samples),
        "counts": _counts(samples),
        "generation": generation or {},
    }
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest


def _counts(samples: List[dict]) -> dict:
    counts: Dict[str, Dict[str, int]] = {}
    for sample in samples:
        bucket = sample.get("_bucket_file", "unfiled")
        counts.setdefault(bucket, {})
        label = sample.get("ground_truth", "?")
        counts[bucket][label] = counts[bucket].get(label, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Version records (§4.17 工程收口)
# ---------------------------------------------------------------------------

VERSIONS_FILE = "VERSIONS.json"


PATH_MAP_PATH = os.path.join(PROJECT_ROOT, "archive", "PATH_MAP.json")


def resolve_path(path: str) -> "tuple":
    """
    (existing path, resolved_from_archive).

    Stage 1 moves the historical material into `archive/legacy_v1_20261015/`;
    the old flat paths stop existing.  A replay that needs one asks here: the
    map answers, and the caller records `resolved_from_archive=True` in its
    report — the archive is a copy being read, and that is part of the result,
    not a hidden assumption.
    """
    absolute = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    if os.path.exists(absolute):
        return absolute, False
    try:
        with open(PATH_MAP_PATH, encoding="utf-8") as handle:
            mapping = json.load(handle)
    except (OSError, ValueError):
        return absolute, False
    relative = os.path.relpath(absolute, PROJECT_ROOT).replace(os.sep, "/")
    for entry in mapping.get("moved", []):
        if relative == entry["from"] or relative.startswith(entry["from"].rstrip("/") + "/"):
            suffix = relative[len(entry["from"].rstrip("/")):].lstrip("/")
            # normpath, not a bare join: for a *file* entry the join with an
            # empty suffix leaves a trailing separator, and `os.path.exists`
            # on a file with a trailing separator is False — which made every
            # single-file mapping report "not from the archive".
            archived = os.path.normpath(
                os.path.join(PROJECT_ROOT, entry["to"].rstrip("/"), suffix))
            return archived, os.path.exists(archived)
    return absolute, False


def batch_versions(experts: Optional[dict] = None, extra: Optional[dict] = None) -> dict:
    """
    The four versions a batch must record to be readable later.

    A batch is only interpretable with the code, the prompt, the expert
    (weight) specs and the calibration table it was produced under.  Old
    batches recorded some of this in scattered reports and none of it in the
    batch directory; this is the single record that travels with the data.
    """
    from datetime import datetime

    from utils.config_fingerprint import calibration_identity, compute, prompt_digest

    versions = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "halting_policy": os.environ.get("FORENSIC_HALTING_POLICY", "v2"),
        "prompt_digest": prompt_digest(),
        "calibration": calibration_identity(),
        "fingerprint": compute(experts or {}),
    }
    if extra:
        versions.update(extra)
    return versions


def write_batch_versions(directory: str, experts: Optional[dict] = None,
                         extra: Optional[dict] = None) -> str:
    """Write `VERSIONS.json` into a batch directory; returns its path."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, VERSIONS_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(batch_versions(experts, extra), handle, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# Frozen batches: a rule the code enforces, not only the docs (§4.17)
# ---------------------------------------------------------------------------

FROZEN_PATH = os.path.join(PROJECT_ROOT, "sft_data", "FROZEN.json")


def frozen_payload() -> dict:
    try:
        with open(FROZEN_PATH, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def frozen_batches() -> Dict[tuple, dict]:
    """(generation, batch) -> entry, for every frozen batch."""
    return {(entry.get("generation"), entry["batch"]): entry
            for entry in frozen_payload().get("frozen", [])}


def generation_register() -> Dict[str, dict]:
    """generation -> its entry in the `generations` index of FROZEN.json."""
    return {entry["generation"]: entry
            for entry in frozen_payload().get("generations", [])
            if entry.get("generation")}


def generation_status(generation: Optional[str]) -> Optional[str]:
    """'active' / 'frozen' / None when the generation is not registered."""
    return (generation_register().get(generation) or {}).get("status")


def assert_generation_is_writable(generation: str, purpose: str,
                                  registered_batch: bool = False) -> None:
    """
    A frozen or unregistered generation must not receive *new* batches.

    The batch-level check alone answers "is *this* batch frozen?", which leaves
    the obvious way around it: keep the frozen generation and pick a name that
    is not in the register.  That is exactly what the two-part identity exists
    to prevent, so the generation is checked as well:

      * **unregistered** — refuse.  Writing material under a generation nobody
        has registered means nobody can say which generation produced it, and a
        typo (`--generation g9` for `g8`) would look like a new generation.
        There is no `--allow` for this: a typo is not a scenario.
      * **frozen, batch not registered** — refuse.  A new batch name inside a
        frozen generation is not a replay of anything, so this has no
        `--allow-frozen` escape either.
      * **frozen, batch registered** — allowed here; the batch-level check
        decides, and `--allow-frozen` is what permits a labelled replay of that
        historical batch.
    """
    register = generation_register()
    entry = register.get(generation)
    if entry is None:
        known = ", ".join(sorted(register)) or "（登记表为空）"
        raise SystemExit(
            f"未登记的代次 {generation}：不能用于{purpose}。"
            f"登记表（sft_data/FROZEN.json 的 generations 索引）里只有：{known}。"
            "新代次请先登记（并同步 sft_data/GENERATIONS.json 的模型身份）——"
            "否则产生的是「没人知道属于哪一代」的材料。")
    if entry.get("status") == "frozen" and not registered_batch:
        raise SystemExit(
            f"代次 {generation} 已冻结（自 {entry.get('since') or '未注明'}："
            f"{entry.get('note') or entry.get('reason') or '未注明理由'}），"
            f"不能用于{purpose}，也**不得在其下新建任何批次**——"
            "冻结身份是 (代次, 批次) 两段，正是为了挡住「换个批次名继续写在冻结代次里」。"
            "新批次请用当前代次（如 --generation g8 --batch b1）；"
            "已登记的历史批次若只是复算，走 --allow-frozen。")


def generation_of(batch_id: str, generation: Optional[str] = None) -> Optional[str]:
    """
    The generation of a batch: what the caller said, else what the register
    knows.  Nothing is guessed from the shape of the name.
    """
    if generation:
        return generation
    for (registered, batch), _entry in frozen_batches().items():
        if batch == batch_id:
            return registered
    return None


def is_frozen(batch_id: str, generation: Optional[str] = None) -> bool:
    return (generation_of(batch_id, generation), batch_id) in frozen_batches()


def assert_not_frozen_batch(batch_id: str, purpose: str, generation: str,
                            allow: bool = False) -> None:
    """
    Refuse to write material into a frozen batch.

    The generation is **required and never guessed**: the batch name no longer
    carries one (`b1` under `sft_data/g8/`), so there is nothing to derive it
    from — a default would be the next "default v2".  Historical ids are
    registered with their generation (`g4_trial`/`g7_b1`/`g7_b2` are all
    `g7`), so a replay passes `generation="g7"` explicitly.

    `--allow-frozen` exists for analysis runs and labels its output, so an
    artifact can never quietly come from a frozen batch.

    The check has **two levels**, because a batch-level check alone leaves the
    obvious way around it: keep the frozen (or unregistered) generation and pick
    a batch name that is not in the register.
    """
    entry = frozen_batches().get((generation, batch_id))
    if entry is None:
        entry = frozen_batches().get((generation_of(batch_id), batch_id))
    # Order matters: the generation decides whether a *new* batch may appear
    # here at all, and a registered frozen batch is the one case where
    # --allow-frozen may still say yes (a labelled replay).
    assert_generation_is_writable(generation, purpose,
                                  registered_batch=entry is not None)
    if entry is None or allow:
        return
    raise SystemExit(
        f"批次 {batch_id}（代次 {generation}）已冻结为历史实验"
        f"（{entry.get('reason', '未注明理由')}），不能用于{purpose}。"
        f"新版请用新代次目录（如 --generation g8 下的 b1）；"
        f"若确需分析该批次，显式加 --allow-frozen，产物会标注 frozen_batch_used=true。")
