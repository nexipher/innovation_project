"""Assemble the final 24-sample SFT candidate manifest from the screened pool.

The selection itself is recorded in `selection.json`; this script only
validates it against the split, the shortlist and the screening CSV, then
writes `manifest.json`.  It refuses duplicate AutoSplice bases, non-train
sources, missing files, hash mismatches and a composition that is not 8/6/6/4.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
DEFAULT_OUT = ROOT / "sft_data/sft_candidates_20261127"
EXPECTED = {
    "autosplice_forged": 8,
    "autosplice_authentic": 6,
    "genimage_generated": 6,
    "genimage_real": 4,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    split = json.loads((out / "split.json").read_text(encoding="utf-8"))
    short = json.loads((out / "shortlist.json").read_text(encoding="utf-8"))
    selection = json.loads((out / "selection.json").read_text(encoding="utf-8"))
    screening = {}
    with (out / "screening/per_sample.csv").open(encoding="utf-8") as handle:
        import csv
        for row in csv.DictReader(handle):
            screening[row["sample_id"]] = row
    pool = {row["variant_id"]: row for row in short["samples"]}
    auto_bases = split["autosplice"]["bases"]
    gen_sources = split["genimage"]["sources"]

    chosen = selection["samples"]
    ids = [row["sample_id"] for row in chosen]
    if len(ids) != 24 or len(set(ids)) != 24:
        raise SystemExit(f"selection must contain 24 unique ids, got {len(ids)}")
    by_kind = collections.Counter()
    used_auto_bases = collections.Counter()
    used_gen_sources = collections.Counter()
    samples = []
    for item in chosen:
        sid = item["sample_id"]
        if sid not in pool:
            raise SystemExit(f"selected sample not in shortlist: {sid}")
        row = dict(pool[sid])
        row["sample_id"] = sid
        if row["split"] != "train":
            raise SystemExit(f"selected sample is not train: {sid}")
        if sid not in screening:
            raise SystemExit(f"selected sample has no screening row: {sid}")
        by_kind[row["kind"]] += 1
        if row["kind"].startswith("autosplice"):
            used_auto_bases[row["source_id"]] += 1
            base = auto_bases[row["source_id"]]
            if base["split"] != "train":
                raise SystemExit(f"base split mismatch: {sid}")
            row["base_variant_count"] = len(base["forged"]) + 1
            row["mask_shared_by_base"] = bool(row.get("mask_path"))
        else:
            used_gen_sources[row["source_id"]] += 1
            source = gen_sources[row["source_id"]]
            if source["split"] != "train":
                raise SystemExit(f"source split mismatch: {sid}")
            if row.get("source_sha256") is None:
                row["source_sha256"] = source["source_sha256"]
        path = ROOT / row["path"]
        if not path.is_file():
            raise SystemExit(f"missing selected image: {path}")
        if sha256_file(path) != row["q95_sha256"]:
            raise SystemExit(f"q95 hash mismatch: {sid}")
        if row.get("mask_path"):
            if sha256_file(ROOT / row["mask_path"]) != row["mask_sha256"]:
                raise SystemExit(f"mask hash mismatch: {sid}")
        screen = screening[sid]
        row.update({
            "selection_role": item["role"],
            "risk_tags": list(item["tags"]),
            "key_case": sid in selection["key_cases"],
            "key_case_rank": (selection["key_cases"].index(sid) + 1
                              if sid in selection["key_cases"] else None),
            "screening_review_only": {
                "probe_score": float(screen["probe_score"]),
                "trufor_score": float(screen["trufor_score"]),
                "trufor_map_high_fraction": float(screen["high_response_fraction"]),
                "trufor_confidence_mean": float(screen["confidence_mean"]),
                "trufor_low_confidence_fraction": float(
                    screen["low_confidence_fraction"]),
                "trufor_iou": (float(screen["iou"]) if screen.get("iou") else None),
            },
            "usage_flags": ["sft_candidate_round1"],
            "usage_roles": ["sft_candidate_review"],
            "no_known_use": False,
            "condition": "jpeg_q95",
            "treatment": "jpeg_q95",
            "encoder": "jpeg",
        })
        samples.append(row)

    if by_kind != collections.Counter(EXPECTED):
        raise SystemExit(f"composition mismatch: {dict(by_kind)} != {EXPECTED}")
    duplicated = {key: count for key, count in used_auto_bases.items() if count > 1}
    if duplicated:
        raise SystemExit(f"AutoSplice base reused: {duplicated}")
    duplicated = {key: count for key, count in used_gen_sources.items() if count > 1}
    if duplicated:
        raise SystemExit(f"GenImage/Real source reused: {duplicated}")
    if len(selection["key_cases"]) not in (6, 7, 8):
        raise SystemExit("key_cases must contain 6-8 ids")

    # Put key cases first in manifest order so the default page order already
    # reflects the review priority.
    rank = {sid: index for index, sid in enumerate(selection["key_cases"])}
    samples.sort(key=lambda row: (rank.get(row["sample_id"], len(rank)),
                                  row["kind"], row["sample_id"]))
    manifest = {
        "schema": "sft_candidate_manifest/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "purpose": ("first-round SFT candidate exemplars; review only, "
                    "not training data until accepted"),
        "seed": selection["seed"],
        "rules": selection["rules"],
        "counts": {
            "total": len(samples),
            "by_kind": dict(by_kind),
            "key_cases": len(selection["key_cases"]),
        },
        "split": {
            "path": "sft_data/sft_candidates_20261127/split.json",
            "all_samples_on": "train",
            "autosplice_unit": "base",
            "genimage_unit": "source_id",
        },
        "samples": samples,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "samples": len(samples),
        "by_kind": dict(by_kind),
        "key_cases": selection["key_cases"],
        "first": samples[0]["sample_id"],
        "last": samples[-1]["sample_id"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
