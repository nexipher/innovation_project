"""Fix the 24-session joint-task acceptance sample (2026-11-07).

8 Real, 8 whole-image AI-generated, 8 local manipulations.  Includes four
AutoSplice real sources with high TruFor map response (real false positives),
and two same-base forged variant pairs, so the reviewer can see whether the
MLLM over-reads a bright map.  Dev/val only; test is never read.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
GEN = ROOT / "dataset/GenImage_q95/eval_dev_manifest_20261102.json"
SUB = ROOT / "sft_data/g8_localization/autosplice_dev_subset_20261102.json"
TRUFOR = ROOT / "sft_data/g8_localization/trufor_autosplice_dev_20261102/report.json"
OUT = ROOT / "sft_data/g8_acceptance/dual_20261107/manifest.json"
GROUPS = ("ADM", "BigGAN", "Glide", "Midjourney", "SD14", "SD15", "VQDM", "Wukong")


def row_from_genimage(row):
    return {
        "sample_id": "genimage__" + row["source_id"].replace("/", "__") + "__q95",
        "variant_id": row["variant_id"],
        "source_id": row["source_id"],
        "family": "GenImage_q95",
        "path": row["path"],
        "source_path": row["source_path"],
        "q95_sha256": row["sha256"],
        "source_sha256": row["source_sha256"],
        "mask_path": None,
        "mask_sha256": None,
        "new_split": row["new_split"],
        "old_split": row["old_split"],
        "no_known_use": row["no_known_use"],
    }


def main():
    gen = json.load(open(GEN, encoding="utf-8"))["inputs"]
    sub = json.load(open(SUB, encoding="utf-8"))
    tr = json.load(open(TRUFOR, encoding="utf-8"))
    samples = []

    gen_real = sorted(
        [row_from_genimage(row) for row in gen
         if row["group"] == "Real" and row["no_known_use"]],
        key=lambda row: row["source_id"])
    gen_real = gen_real[:4]
    for row in gen_real:
        row.update({"group": "Real", "ground_truth": "Real",
                    "task_truth": {"whole_image_generation": "Real",
                                   "local_manipulation": "Not tampered"},
                    "risk_tags": []})
        samples.append(row)

    # Four real AutoSplice sources with the highest TruFor high-response area.
    tr_sources = {source["source_id"]: source
                  for source in tr["aggregate"]["sources"] if source.get("authentic")}
    high_real = sorted(
        (source for source in tr_sources.values()
         if source["variant_count"] == 0),
        key=lambda source: -source["authentic"]["map_high_fraction"])[:4]
    sub_inputs = {row["variant_id"]: row for row in sub["inputs"]}
    for source in high_real:
        row = sub_inputs[source["authentic"]["variant_id"]]
        samples.append({
            "sample_id": "autosplice__" + row["source_id"] + "__authentic__q95",
            "variant_id": row["variant_id"], "source_id": row["source_id"],
            "family": "AutoSplice_q95", "group": "Real",
            "path": row["path"], "source_path": row["source_path"],
            "q95_sha256": row["sha256"], "source_sha256": row["source_sha256"],
            "mask_path": None, "mask_sha256": None,
            "new_split": row["split"], "old_split": "not_in_old",
            "no_known_use": row.get("no_known_use"),
            "ground_truth": "Real",
            "task_truth": {"whole_image_generation": "Real",
                           "local_manipulation": "Not tampered"},
            "risk_tags": ["trufor_real_high_response"],
        })

    # One whole-image fake from each GenImage generator, no-known-use.
    for group in GROUPS:
        candidates = sorted(
            [row_from_genimage(row) for row in gen
             if row["group"] == group and row["label"] == "Fake"
             and row["no_known_use"]],
            key=lambda row: row["source_id"])
        if not candidates:
            raise SystemExit(f"no no-known-use fake for {group}")
        row = candidates[0]
        row.update({"group": group, "ground_truth": "Fake",
                    "task_truth": {"whole_image_generation": "AI-generated",
                                   "local_manipulation": "Not applicable"},
                    "risk_tags": ["whole_image_generated"]})
        samples.append(row)

    # Eight local forged inputs: two same-base pairs plus four varied bases.
    forged = [row for row in sub["inputs"] if row["label"] == "Fake"]
    by_source = {}
    for row in forged:
        by_source.setdefault(row["source_id"], []).append(row)
    chosen = []
    for sid in ("39890", "41187"):
        variants = sorted(by_source.get(sid, []), key=lambda row: int(row["variant"]))
        chosen.extend(variants[:2])
    remaining = sorted(
        [row for row in forged if row["source_id"] not in {"39890", "41187"}],
        key=lambda row: (row.get("mask_sha256") or "", row["source_id"],
                         int(row["variant"])))
    chosen.extend(remaining[:4])
    if len(chosen) != 8:
        raise SystemExit(f"chosen forged {len(chosen)}")
    for row in chosen:
        samples.append({
            "sample_id": row["variant_id"], "variant_id": row["variant_id"],
            "source_id": row["source_id"], "family": "AutoSplice_q95",
            "group": "AutoSplice", "path": row["path"],
            "source_path": row["source_path"], "q95_sha256": row["sha256"],
            "source_sha256": row["source_sha256"],
            "mask_path": row.get("mask_path"), "mask_sha256": row.get("mask_sha256"),
            "new_split": row["split"], "old_split": "not_in_old",
            "no_known_use": row.get("no_known_use"), "ground_truth": "Fake",
            "task_truth": {"whole_image_generation": "Real",
                           "local_manipulation": "Tampered"},
            "risk_tags": ["local_edit"] + (
                ["same_base_multi_variant"] if len(by_source[row["source_id"]]) > 1 else []),
        })

    if len(samples) != 24:
        raise SystemExit(f"expected 24 samples, got {len(samples)}")
    ids = [row["sample_id"] for row in samples]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate sample_id")
    for row in samples:
        path = ROOT / row["path"]
        if not path.exists():
            raise SystemExit(f"missing {path}")
        if row.get("mask_path") and not (ROOT / row["mask_path"]).exists():
            raise SystemExit(f"missing {row['mask_path']}")

    payload = {
        "schema": "dual_acceptance_manifest/1",
        "created_at": "2026-11-07",
        "purpose": "human acceptance of PROBE+TruFor joint task; not training data",
        "rules": {
            "tools": ["probe", "trufor"],
            "splits": "dev/val only; test not read",
            "counts": {"total": 24, "real": 8, "whole_generated": 8,
                       "local_manipulation": 8},
            "mask": "AutoSplice base-level shared mask",
        },
        "counts": {
            "total": len(samples),
            "by_family": dict(Counter(row["family"] for row in samples)),
            "by_truth": {
                "whole_image_generation": dict(Counter(
                    row["task_truth"]["whole_image_generation"] for row in samples)),
                "local_manipulation": dict(Counter(
                    row["task_truth"]["local_manipulation"] for row in samples)),
            },
            "risk_tags": dict(Counter(tag for row in samples for tag in row["risk_tags"])),
        },
        "samples": samples,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps(payload["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
