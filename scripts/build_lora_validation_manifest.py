"""Build ~30 source-disjoint validation inputs for the 24-sample LoRA trial.

Uses only the validation side of `sft_candidates_20261127/split.json`, so every
AutoSplice base and GenImage/Real source is disjoint from the 24 training
trajectories (which are all from the train side).  Composition:
  10 AutoSplice forged, 5 AutoSplice authentic, 10 GenImage generated,
  5 GenImage Real.
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
SMALL = ROOT / "sft_data/lora_small_20261128"
SPLIT = ROOT / "sft_data/sft_candidates_20261127/split.json"
TRAIN_RECORDS = ROOT / "sft_data/sft_candidates_20261127/records.json"
OUT = SMALL / "validation_manifest.json"
SEED = 20261128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_records = json.loads(TRAIN_RECORDS.read_text(encoding="utf-8"))
    train_sources = {row["source_id"] for row in train_records}
    rng = random.Random(SEED)

    forged_val = []
    authentic_val = []
    for base, entry in split["autosplice"]["bases"].items():
        if entry["split"] != "val":
            continue
        if entry["forged"]:
            forged_val.append(base)
        else:
            authentic_val.append(base)
    if len(forged_val) < 10 or len(authentic_val) < 5:
        raise SystemExit(f"not enough AutoSplice val bases: {len(forged_val)} / {len(authentic_val)}")
    forged_bases = sorted(rng.sample(sorted(forged_val), 10))
    authentic_bases = sorted(rng.sample(sorted(authentic_val), 5))

    gen_by_group = collections.defaultdict(list)
    real_val = []
    for sid, entry in split["genimage"]["sources"].items():
        if entry["split"] != "val":
            continue
        if entry["group"] == "Real":
            real_val.append(sid)
        else:
            gen_by_group[entry["group"]].append(sid)
    groups = sorted(gen_by_group)
    generated = []
    for group in groups:
        generated.append(rng.choice(sorted(gen_by_group[group])))
    candidates = [sid for group in groups for sid in sorted(gen_by_group[group])
                  if sid not in generated]
    generated.extend(rng.sample(candidates, 2))
    generated = sorted(set(generated))
    if len(generated) != 10:
        raise SystemExit(f"generated selection is {len(generated)}, expected 10")
    real_sources = sorted(rng.sample(sorted(real_val), 5))

    samples = []
    for base in forged_bases:
        entry = split["autosplice"]["bases"][base]
        forged = sorted(entry["forged"], key=lambda item: item["variant"])[0]
        samples.append({
            "sample_id": f"val__autosplice__{base}__{forged['variant']}__q95",
            "source_id": base, "family": "AutoSplice_q95",
            "group": "AutoSplice_Forged", "source_role": "autosplice_forged",
            "path": forged["path"], "source_path": forged["source_path"],
            "q95_sha256": forged["q95_sha256"], "bytes": forged["bytes"],
            "width": forged["width"], "height": forged["height"],
            "ground_truth": "Fake",
            "task_truth": {"whole_image_generation": "Real",
                           "local_manipulation": "Tampered"},
            "split": "val", "condition": "jpeg_q95", "treatment": "jpeg_q95",
        })
    for base in authentic_bases:
        entry = split["autosplice"]["bases"][base]
        auth = entry["authentic"]
        samples.append({
            "sample_id": f"val__autosplice__{base}__authentic__q95",
            "source_id": base, "family": "AutoSplice_q95",
            "group": "AutoSplice_Authentic", "source_role": "autosplice_authentic",
            "path": auth["path"], "source_path": auth["source_path"],
            "q95_sha256": auth["q95_sha256"], "bytes": auth["bytes"],
            "width": auth["width"], "height": auth["height"],
            "ground_truth": "Real",
            "task_truth": {"whole_image_generation": "Real",
                           "local_manipulation": "Not tampered"},
            "split": "val", "condition": "jpeg_q95", "treatment": "jpeg_q95",
        })
    for sid in generated:
        entry = split["genimage"]["sources"][sid]
        samples.append({
            "sample_id": f"val__genimage__{sid.replace('/', '__')}__q95",
            "source_id": sid, "family": "GenImage_q95",
            "group": entry["group"], "source_role": "genimage_generated",
            "path": entry["path"], "source_path": entry["source_path"],
            "q95_sha256": entry["q95_sha256"],
            "source_sha256": entry["source_sha256"],
            "bytes": entry.get("bytes"), "width": entry.get("width"),
            "height": entry.get("height"),
            "ground_truth": "Fake",
            "task_truth": {"whole_image_generation": "AI-generated",
                           "local_manipulation": "Not applicable"},
            "split": "val", "condition": "jpeg_q95", "treatment": "jpeg_q95",
        })
    for sid in real_sources:
        entry = split["genimage"]["sources"][sid]
        samples.append({
            "sample_id": f"val__genimage__{sid.replace('/', '__')}__q95",
            "source_id": sid, "family": "GenImage_q95",
            "group": "Real", "source_role": "genimage_real",
            "path": entry["path"], "source_path": entry["source_path"],
            "q95_sha256": entry["q95_sha256"],
            "source_sha256": entry["source_sha256"],
            "bytes": entry.get("bytes"), "width": entry.get("width"),
            "height": entry.get("height"),
            "ground_truth": "Real",
            "task_truth": {"whole_image_generation": "Real",
                           "local_manipulation": "Not tampered"},
            "split": "val", "condition": "jpeg_q95", "treatment": "jpeg_q95",
        })

    if len(samples) != 30:
        raise SystemExit(f"expected 30 validation samples, got {len(samples)}")
    source_ids = [row["source_id"] for row in samples]
    if len(source_ids) != len(set(source_ids)):
        raise SystemExit("duplicate validation source")
    overlap = sorted(set(source_ids) & train_sources)
    if overlap:
        raise SystemExit(f"validation overlaps training sources: {overlap[:5]}")
    for row in samples:
        path = ROOT / row["path"]
        if not path.is_file():
            raise SystemExit(f"missing validation image: {path}")
        if sha256_file(path) != row["q95_sha256"]:
            raise SystemExit(f"q95 hash mismatch: {row['sample_id']}")
    payload = {
        "schema": "lora_validation_manifest/1",
        "created_at": "2026-11-28",
        "seed": SEED,
        "purpose": "30 source-disjoint validation images for base-vs-LoRA autonomous comparison",
        "counts": dict(collections.Counter(row["group"] for row in samples)),
        "samples": samples,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    print(json.dumps({"out": str(OUT), "samples": len(samples),
                      "counts": payload["counts"],
                      "groups": dict(collections.Counter(row["group"] for row in samples))},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
