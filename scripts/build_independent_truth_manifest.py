"""Build the 2026-11-28 independent-local truth view for the 30 validation images.

The runtime manifest keeps the historical labels.  This view makes the local
question independent of the whole-image label:

* AutoSplice forged -> local Tampered;
* AutoSplice authentic / Real -> local Not tampered;
* GenImage generated -> local Not tampered only because the GenImage
  construction manifest records a generator-only image and no local-edit step;
  the old Not applicable value is retained in `local_truth_old` for audit.

It writes a new manifest and a provenance CSV.  No images or historical
records are changed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
SMALL = ROOT / "sft_data/lora_small_20261128"
SOURCE = SMALL / "validation_manifest.json"
GENIMAGE = ROOT / "dataset/GenImage_q95/manifest.json"
OUT_MANIFEST = SMALL / "validation_manifest_independent_20261128.json"
OUT_CSV = SMALL / "analysis/rule_update_20261128/truth_provenance.csv"


def main() -> int:
    manifest = json.loads(SOURCE.read_text(encoding="utf-8"))
    gen_inputs = {
        row["source_id"]: row
        for row in json.loads(GENIMAGE.read_text(encoding="utf-8"))["inputs"]
    }
    rows = []
    for sample in manifest["samples"]:
        row = dict(sample)
        row["task_truth"] = dict(sample["task_truth"])
        old_local = row["task_truth"].get("local_manipulation")
        row["local_truth_old"] = old_local
        group = sample["group"]
        if group == "AutoSplice_Forged":
            local_truth = "Tampered"
            basis = "AutoSplice forged base + mask; local edit is the target"
            local_group = "forged_positive"
            construction = sample.get("source_id", "")
        elif group in ("AutoSplice_Authentic", "Real"):
            local_truth = "Not tampered"
            basis = "authentic/real pool; no local-edit construction record"
            local_group = "real_negative"
            construction = ""
        else:
            source_id = sample["source_id"]
            entry = gen_inputs.get(source_id)
            if entry is None:
                raise SystemExit(f"missing GenImage construction record: {source_id}")
            if entry.get("generator") not in (None, group):
                raise SystemExit(
                    f"generator mismatch for {source_id}: {entry.get('generator')} != {group}")
            local_truth = "Not tampered"
            basis = ("GenImage manifest: generator-only q95 derivative; no local-edit "
                     "step or local mask recorded")
            local_group = "generated_negative"
            construction = entry.get("source_path", "")
            row["construction_generator"] = entry.get("generator")
            row["construction_source_path"] = entry.get("source_path")
            row["construction_treatment"] = entry.get("treatment")
        row["task_truth"]["local_manipulation"] = local_truth
        row["local_truth_group"] = local_group
        row["local_truth_basis"] = basis
        row["construction_path"] = construction
        rows.append(row)

    OUT_MANIFEST.write_text(
        json.dumps({**manifest, "samples": rows,
                    "purpose": "independent local truth view for rule 2026-11-28"},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        fields = ["sample_id", "source_id", "group", "local_truth_group",
                  "local_truth_old", "task_truth_json", "local_truth_basis",
                  "construction_generator", "construction_source_path",
                  "construction_treatment", "construction_path"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row.get(key) for key in fields
                   if key not in ("task_truth_json",)},
                "task_truth_json": json.dumps(row["task_truth"], ensure_ascii=False),
            })
    from collections import Counter
    print(json.dumps({
        "manifest": str(OUT_MANIFEST),
        "provenance": str(OUT_CSV),
        "counts": dict(Counter(row["local_truth_group"] for row in rows)),
        "old_local": dict(Counter(row["local_truth_old"] for row in rows)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
