"""Build the first-round SFT candidate split and screening shortlist.

Sources are outside the 200-image development evaluation:
  * AutoSplice: full dataset, every base NOT in the 100-base dev subset.
    Base is the split unit; every authentic/forged variant inherits it.
  * GenImage generated + Real: only inputs marked `no_known_use=True` in the
    2026-11-02 dev/val manifests, excluding the 200 eval sources and the
    previous 24-session control sources.  source_id is the split unit.

The split is 90/10 train/val, deterministic per stratum.  Candidates for the
24 exemplars are drawn from the train side only.  A larger shortlist is written
for independent PROBE/TruFor screening; screening picks the final 24.

No model is loaded here.  Ground truth and masks are recorded for review only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path("/root/autodl-tmp/innovation_project")
SEED = 20261127
DEV_SUBSET = ROOT / "sft_data/g8_localization/autosplice_dev_subset_20261102.json"
DEV_EVAL = ROOT / "sft_data/g8_dev_eval_20261110/manifest.json"
PREVIOUS_24 = ROOT / "sft_data/g8_acceptance/dual_20261107/manifest.json"
GEN_MANIFESTS = (
    ROOT / "dataset/GenImage_q95/eval_dev_manifest_20261102.json",
    ROOT / "dataset/GenImage_q95/eval_val_manifest_20261102.json",
)
AUTOSPLICE = ROOT / "dataset/AutoSplice"
AUTOSPLICE_Q95 = ROOT / "dataset/AutoSplice_q95"
OUT_DIR = ROOT / "sft_data/sft_candidates_20261127"

SHORTLIST_FORGED_BASES = 40
SHORTLIST_AUTHENTIC_BASES = 30
SHORTLIST_GEN_PER_GROUP = 2
SHORTLIST_REAL = 10


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def image_meta(path: Path) -> dict:
    with Image.open(path) as image:
        width, height = image.size
    return {"bytes": path.stat().st_size, "width": width, "height": height,
            "sha256": sha256_file(path)}


def split_keys(keys, stratum: str, train_ratio: float = 0.9) -> dict:
    """Exact-count deterministic split inside one stratum."""
    ordered = sorted(keys)
    rng = random.Random(f"{SEED}:{stratum}")
    rng.shuffle(ordered)
    cut = int(round(len(ordered) * train_ratio))
    return {key: ("train" if index < cut else "val")
            for index, key in enumerate(ordered)}


def autosplice_pool() -> tuple[dict, dict]:
    dev = json.loads(DEV_SUBSET.read_text(encoding="utf-8"))
    dev_bases = {str(base["source_id"]) for base in dev["bases"]}

    forged = defaultdict(list)
    for path in sorted((AUTOSPLICE / "Forged_JPEG100").glob("*.jpg")):
        match = re.match(r"(.+)_(\d+)\.jpg$", path.name)
        if not match:
            raise SystemExit(f"unexpected forged filename: {path.name}")
        forged[match.group(1)].append(match.group(2))
    authentic = {path.stem: path for path in sorted(
        (AUTOSPLICE / "Authentic").glob("*.jpg"))}

    missing = sorted(set(forged) - set(authentic))
    if missing:
        raise SystemExit(f"forged bases without authentic image: {missing[:5]}")
    # Every forged-capable base outside the dev subset must have a mask.
    masks = {path.stem.removesuffix("_mask"): path for path in
             (AUTOSPLICE / "Mask").glob("*_mask.png")}
    missing_masks = sorted((set(forged) - dev_bases) - set(masks))
    if missing_masks:
        raise SystemExit(f"forged bases without mask: {missing_masks[:5]}")

    forged_bases = sorted(set(forged) - dev_bases)
    authentic_only = sorted((set(authentic) - set(forged)) - dev_bases)
    splits = {}
    splits.update(split_keys(forged_bases, "autosplice:forged-capable"))
    splits.update(split_keys(authentic_only, "autosplice:authentic-only"))
    return {"forged": forged, "authentic": authentic, "masks": masks,
            "forged_bases": forged_bases, "authentic_only": authentic_only,
            "splits": splits}


def genimage_pool(excluded_sources: set[str]) -> list[dict]:
    rows = []
    seen = set()
    for manifest_path in GEN_MANIFESTS:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in payload["inputs"]:
            if not row.get("no_known_use"):
                continue
            if row["source_id"] in excluded_sources:
                continue
            if row["source_id"] in seen:
                raise SystemExit(f"duplicate GenImage source: {row['source_id']}")
            seen.add(row["source_id"])
            rows.append(dict(row, _pool_split=payload["split"]))
    return rows


def autosplice_entry(pool: dict, base: str) -> dict:
    authentic = pool["authentic"][base]
    forged = []
    if base in pool["forged"]:
        for variant in sorted(pool["forged"][base], key=int):
            path = AUTOSPLICE_Q95 / "Forged_JPEG100" / f"{base}_{variant}.jpg"
            if not path.is_file():
                raise SystemExit(f"missing q95 forged: {path}")
            meta = image_meta(path)
            forged.append({
                "variant": variant,
                "variant_id": f"autosplice__{base}__{variant}__q95",
                "path": rel(path),
                "source_path": rel(AUTOSPLICE / "Forged_JPEG100" /
                                   f"{base}_{variant}.jpg"),
                "q95_sha256": meta["sha256"],
                "bytes": meta["bytes"],
                "width": meta["width"],
                "height": meta["height"],
            })
    auth_q95 = AUTOSPLICE_Q95 / "Authentic" / f"{base}.jpg"
    if not auth_q95.is_file():
        raise SystemExit(f"missing q95 authentic: {auth_q95}")
    auth_meta = image_meta(auth_q95)
    entry = {
        "base": base,
        "split": pool["splits"][base],
        "authentic": {
            "variant_id": f"autosplice__{base}__authentic__q95",
            "path": rel(auth_q95),
            "source_path": rel(authentic),
            "q95_sha256": auth_meta["sha256"],
            "bytes": auth_meta["bytes"],
            "width": auth_meta["width"],
            "height": auth_meta["height"],
        },
        "forged": forged,
    }
    if base in pool["masks"]:
        mask = pool["masks"][base]
        entry["mask"] = {"path": rel(mask), "sha256": sha256_file(mask)}
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args()
    out = Path(args.out)

    eval_manifest = json.loads(DEV_EVAL.read_text(encoding="utf-8"))
    eval_sources = {row["source_id"] for row in eval_manifest["samples"]
                    if row["family"] == "GenImage_q95"}
    eval_auto = {str(row["source_id"]) for row in eval_manifest["samples"]
                 if row["family"] == "AutoSplice_q95"}
    previous = json.loads(PREVIOUS_24.read_text(encoding="utf-8"))
    previous_sources = {row["source_id"] for row in previous["samples"]
                        if row["family"] == "GenImage_q95"}
    excluded = eval_sources | previous_sources

    pool = autosplice_pool()
    # The 200 eval used only bases inside the dev subset, so the two sets are
    # disjoint by construction.  Assert rather than assume.
    if eval_auto & set(pool["forged_bases"]):
        raise SystemExit("200 eval AutoSplice bases leak into the SFT pool")

    gen_rows = genimage_pool(excluded)
    groups = defaultdict(list)
    for row in gen_rows:
        groups[row["group"]].append(row)
    gen_splits = {}
    for group, rows in sorted(groups.items()):
        gen_splits.update(split_keys([row["source_id"] for row in rows],
                                     f"genimage:{group}"))
    gen_by_source = {row["source_id"]: dict(row, split=gen_splits[row["source_id"]])
                     for row in gen_rows}

    auto_entries = {base: autosplice_entry(pool, base) for base in
                    sorted(set(pool["forged_bases"]) | set(pool["authentic_only"]))}
    auto_train_forged = [b for b in pool["forged_bases"]
                         if auto_entries[b]["split"] == "train"]
    auto_train_auth = [b for b in pool["authentic_only"]
                       if auto_entries[b]["split"] == "train"]
    auto_val_forged = [b for b in pool["forged_bases"]
                       if auto_entries[b]["split"] == "val"]
    auto_val_auth = [b for b in pool["authentic_only"]
                     if auto_entries[b]["split"] == "val"]
    gen_train = {group: sorted(row["source_id"] for row in rows
                               if gen_by_source[row["source_id"]]["split"] == "train")
                 for group, rows in groups.items()}
    gen_val = {group: sorted(row["source_id"] for row in rows
                             if gen_by_source[row["source_id"]]["split"] == "val")
               for group, rows in groups.items()}

    rng = random.Random(f"{SEED}:shortlist")
    short_forged = sorted(rng.sample(auto_train_forged, SHORTLIST_FORGED_BASES))
    short_auth = sorted(rng.sample(auto_train_auth, SHORTLIST_AUTHENTIC_BASES))
    short_gen = []
    for group in sorted(gen_train):
        if group == "Real":
            continue
        if len(gen_train[group]) < SHORTLIST_GEN_PER_GROUP:
            raise SystemExit(f"not enough train GenImage sources for {group}")
        short_gen.extend(sorted(rng.sample(gen_train[group],
                                           SHORTLIST_GEN_PER_GROUP)))
    real_train = gen_train.get("Real") or []
    if len(real_train) < SHORTLIST_REAL:
        raise SystemExit("not enough train Real sources")
    short_real = sorted(rng.sample(real_train, SHORTLIST_REAL))

    split_payload = {
        "schema": "sft_candidate_split/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "purpose": "first-round SFT candidate split; candidates from train only",
        "rules": {
            "autosplice_unit": "base; authentic and all forged variants inherit the base split",
            "genimage_unit": "source_id; q95 variant inherits the source split",
            "ratio": "90/10 train/val per stratum, exact-count deterministic shuffle",
            "autosplice_pool": "full AutoSplice minus the 100-base dev subset",
            "genimage_pool": "2026-11-02 dev/val manifests, no_known_use=True only",
            "exclusions": "200 eval sources, previous 24 control sources, no_known_use=False",
            "test": "not read, not written",
        },
        "exclusions": {
            "eval_genimage_sources": sorted(eval_sources),
            "eval_autosplice_bases": sorted(eval_auto),
            "previous_24_genimage_sources": sorted(previous_sources),
        },
        "autosplice": {
            "counts": {
                "forged_capable": len(pool["forged_bases"]),
                "authentic_only": len(pool["authentic_only"]),
                "train_forged_capable": len(auto_train_forged),
                "train_authentic_only": len(auto_train_auth),
                "val_forged_capable": len(auto_val_forged),
                "val_authentic_only": len(auto_val_auth),
            },
            "bases": auto_entries,
        },
        "genimage": {
            "counts": {group: {"train": len(gen_train[group]),
                               "val": len(gen_val[group])}
                       for group in sorted(groups)},
            "sources": {source_id: {
                "split": row["split"],
                "group": row["group"],
                "path": row["path"],
                "source_path": row["source_path"],
                "q95_sha256": row["sha256"],
                "source_sha256": row["source_sha256"],
                "bytes": row.get("bytes"),
                "width": row.get("width"),
                "height": row.get("height"),
            } for source_id, row in sorted(gen_by_source.items())},
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.json").write_text(
        json.dumps(split_payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")

    shortlist = []
    for base in short_forged:
        entry = auto_entries[base]
        for forged in entry["forged"]:
            shortlist.append({
                "kind": "autosplice_forged",
                "source_id": base,
                "variant_id": forged["variant_id"],
                "variant": forged["variant"],
                "path": forged["path"],
                "source_path": forged["source_path"],
                "q95_sha256": forged["q95_sha256"],
                "mask_path": entry["mask"]["path"],
                "mask_sha256": entry["mask"]["sha256"],
                "split": entry["split"],
                "group": "AutoSplice_Forged",
                "ground_truth": "Fake",
                "task_truth": {"whole_image_generation": "Real",
                               "local_manipulation": "Tampered"},
            })
    for base in short_auth:
        entry = auto_entries[base]
        auth = entry["authentic"]
        shortlist.append({
            "kind": "autosplice_authentic",
            "source_id": base,
            "variant_id": auth["variant_id"],
            "variant": "authentic",
            "path": auth["path"],
            "source_path": auth["source_path"],
            "q95_sha256": auth["q95_sha256"],
            "mask_path": None,
            "mask_sha256": None,
            "split": entry["split"],
            "group": "AutoSplice_Authentic",
            "ground_truth": "Real",
            "task_truth": {"whole_image_generation": "Real",
                           "local_manipulation": "Not tampered"},
        })
    for source_id in short_gen:
        row = gen_by_source[source_id]
        shortlist.append({
            "kind": "genimage_generated",
            "source_id": source_id,
            "variant_id": row["variant_id"],
            "variant": "q95",
            "path": row["path"],
            "source_path": row["source_path"],
            "q95_sha256": row["sha256"],
            "source_sha256": row["source_sha256"],
            "mask_path": None,
            "mask_sha256": None,
            "split": row["split"],
            "group": row["group"],
            "ground_truth": "Fake",
            "task_truth": {"whole_image_generation": "AI-generated",
                           "local_manipulation": "Not applicable"},
        })
    for source_id in short_real:
        row = gen_by_source[source_id]
        shortlist.append({
            "kind": "genimage_real",
            "source_id": source_id,
            "variant_id": row["variant_id"],
            "variant": "q95",
            "path": row["path"],
            "source_path": row["source_path"],
            "q95_sha256": row["sha256"],
            "source_sha256": row["source_sha256"],
            "mask_path": None,
            "mask_sha256": None,
            "split": row["split"],
            "group": "Real",
            "ground_truth": "Real",
            "task_truth": {"whole_image_generation": "Real",
                           "local_manipulation": "Not tampered"},
        })
    # All shortlist rows must come from train and must be hash-verifiable.
    for row in shortlist:
        if row["split"] != "train":
            raise SystemExit(f"shortlist row is not train: {row['variant_id']}")
        path = ROOT / row["path"]
        if not path.is_file():
            raise SystemExit(f"missing shortlist image: {path}")
        if sha256_file(path) != row["q95_sha256"]:
            raise SystemExit(f"q95 hash mismatch: {row['variant_id']}")
        if row.get("source_sha256"):
            if sha256_file(ROOT / row["source_path"]) != row["source_sha256"]:
                raise SystemExit(f"source hash mismatch: {row['variant_id']}")
        if row.get("mask_path"):
            if sha256_file(ROOT / row["mask_path"]) != row["mask_sha256"]:
                raise SystemExit(f"mask hash mismatch: {row['variant_id']}")

    shortlist_payload = {
        "schema": "sft_candidate_shortlist/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "purpose": "pre-screen pool; not the final 24",
        "counts": dict(Counter(row["kind"] for row in shortlist)),
        "samples": shortlist,
    }
    (out / "shortlist.json").write_text(
        json.dumps(shortlist_payload, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "split_counts": {
            "autosplice": split_payload["autosplice"]["counts"],
            "genimage": split_payload["genimage"]["counts"],
        },
        "shortlist": len(shortlist),
        "shortlist_by_kind": dict(Counter(row["kind"] for row in shortlist)),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
