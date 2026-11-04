"""Fix the AutoSplice q95 dev subset for localization evaluation.

100 bases / 220 q95 inputs, chosen before any model run:

* 30 authentic-only bases (30 Real);
* 35 bases with one forged variant (35 Real + 35 Fake);
* 20 bases with two forged variants (20 Real + 40 Fake);
* 15 bases with three forged variants (15 Real + 45 Fake).

Forged bases are stratified by mask-area tertile.  Authentic-only bases are
stratified by image-size tertile.  Every selected base contributes its authentic
image and all of its forged variants; the mask is the base-level shared mask.

The script only reads manifests/images and writes one JSON + one Markdown file.
It does not run a model.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/root/autodl-tmp/innovation_project")
PAIRING = ROOT / "dataset/AutoSplice_q95/pairing_manifest.json"
INPUTS = ROOT / "dataset/AutoSplice_q95/inputs_manifest.json"
SPLITS = ROOT / "dataset/AutoSplice_q95/split_manifest.json"
OUT = ROOT / "sft_data/g8_localization/autosplice_dev_subset_20261102.json"
OUT_MD = ROOT / "docs/development/AUTOSPLICE_LOCALIZATION_SUBSET_20261102.md"
SEED = 20261102
QUOTAS = {
    "authentic_only": 30,
    "forged_single": 35,
    "forged_double": 20,
    "forged_triple": 15,
}
# Balanced across the three mask-area bins; sums match QUOTAS.
AREA_QUOTAS = {
    "forged_single": {"small": 12, "mid": 12, "large": 11},
    "forged_double": {"small": 7, "mid": 7, "large": 6},
    "forged_triple": {"small": 5, "mid": 5, "large": 5},
}
SIZE_QUOTAS = {"small": 10, "mid": 10, "large": 10}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def q95_rows_by_source(rows):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row["source_id"]].append(row)
    return by_source


def pick(candidates, n, rng):
    candidates = sorted(candidates, key=lambda source: source["source_id"])
    rng.shuffle(candidates)
    return candidates[:n]


def main():
    pairing = json.load(open(PAIRING, encoding="utf-8"))
    inputs = json.load(open(INPUTS, encoding="utf-8"))["inputs"]
    splits = json.load(open(SPLITS, encoding="utf-8"))["sources"]
    q95 = q95_rows_by_source(inputs)

    dev = [source for source in pairing["sources"]
           if splits.get(source["source_id"]) == "dev"]
    if not dev:
        raise SystemExit("no dev AutoSplice bases")
    forged = [source for source in dev if source["variant_count"] > 0]
    authentic_only = [source for source in dev if source["variant_count"] == 0]
    if len(forged) < sum(v for k, v in QUOTAS.items() if k.startswith("forged")):
        raise SystemExit("not enough forged dev bases")
    if len(authentic_only) < QUOTAS["authentic_only"]:
        raise SystemExit("not enough authentic-only dev bases")

    areas = np.asarray([source["mask"]["area_fraction"] for source in forged],
                       dtype=np.float64)
    area_cuts = np.percentile(areas, [100 / 3, 200 / 3])
    for source in forged:
        area = float(source["mask"]["area_fraction"])
        source["_area_bin"] = ("small" if area <= area_cuts[0]
                               else "mid" if area <= area_cuts[1] else "large")

    sizes = np.asarray([max(source["authentic"]["width"],
                            source["authentic"]["height"])
                        for source in authentic_only], dtype=np.float64)
    size_cuts = np.percentile(sizes, [100 / 3, 200 / 3])
    for source in authentic_only:
        size = max(source["authentic"]["width"], source["authentic"]["height"])
        source["_size_bin"] = ("small" if size <= size_cuts[0]
                               else "mid" if size <= size_cuts[1] else "large")

    rng = random.Random(SEED)
    selected = []
    notes = []

    # Authentic-only: 10 per image-size bin.
    for bin_name, quota in SIZE_QUOTAS.items():
        pool = [source for source in authentic_only
                if source["_size_bin"] == bin_name and source not in selected]
        chosen = pick(pool, quota, rng)
        if len(chosen) < quota:
            notes.append(f"authentic-only {bin_name}: only {len(chosen)}/{quota}")
        selected.extend(chosen)

    # Forged: variant-count strata x mask-area tertiles.
    for role, variant_count in (("forged_single", 1), ("forged_double", 2),
                                ("forged_triple", 3)):
        for bin_name, quota in AREA_QUOTAS[role].items():
            pool = [source for source in forged
                    if source["variant_count"] == variant_count
                    and source["_area_bin"] == bin_name
                    and source not in selected]
            chosen = pick(pool, quota, rng)
            if len(chosen) < quota:
                notes.append(f"{role} {bin_name}: only {len(chosen)}/{quota}")
            selected.extend(chosen)

    # Fill any shortfall from remaining compatible bases, deterministically.
    if len(selected) < sum(QUOTAS.values()):
        remaining = [source for source in dev if source not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[:sum(QUOTAS.values()) - len(selected)])
        notes.append("filled shortfall from remaining dev bases")
    if len(selected) != sum(QUOTAS.values()):
        raise SystemExit(f"selected {len(selected)} bases, expected "
                         f"{sum(QUOTAS.values())}")

    selected = sorted(selected, key=lambda source: source["source_id"])
    roles = Counter()
    for source in selected:
        vc = source["variant_count"]
        roles["authentic_only" if vc == 0 else
              "forged_single" if vc == 1 else
              "forged_double" if vc == 2 else "forged_triple"] += 1

    bases = []
    selected_inputs = []
    for source in selected:
        sid = source["source_id"]
        vc = source["variant_count"]
        role = ("authentic_only" if vc == 0 else
                "forged_single" if vc == 1 else
                "forged_double" if vc == 2 else "forged_triple")
        rows = q95.get(sid, [])
        if not rows:
            raise SystemExit(f"no q95 rows for {sid}")
        if {row["split"] for row in rows} != {"dev"}:
            raise SystemExit(f"source {sid} crosses partitions: "
                             f"{sorted({row['split'] for row in rows})}")
        real = [row for row in rows if row["label"] == "Real"]
        fake = sorted([row for row in rows if row["label"] == "Fake"],
                      key=lambda row: int(row["variant"]))
        if len(real) != 1 or len(fake) != vc:
            raise SystemExit(f"source {sid}: expected 1 real / {vc} fake, "
                             f"got {len(real)}/{len(fake)}")
        mask = source.get("mask") or {}
        base = {
            "source_id": sid,
            "variant_count": vc,
            "selection_role": role,
            "area_bin": source.get("_area_bin") if vc else None,
            "size_bin": source.get("_size_bin") if vc == 0 else None,
            "mask_shared_by_base": bool(mask),
            "mask": {
                "path": mask.get("path"),
                "sha256": mask.get("sha256"),
                "bytes": mask.get("bytes"),
                "width": mask.get("width"),
                "height": mask.get("height"),
                "area_fraction": mask.get("area_fraction"),
                "bbox": mask.get("bbox"),
            } if mask else None,
            "caption": source.get("caption"),
            "authentic": {
                "variant_id": real[0]["variant_id"],
                "path": real[0]["path"],
                "sha256": real[0]["sha256"],
                "source_path": real[0]["source_path"],
                "source_sha256": real[0]["source_sha256"],
                "width": real[0]["width"],
                "height": real[0]["height"],
                "split": real[0]["split"],
            },
            "forged": [{
                "variant": row["variant"],
                "variant_id": row["variant_id"],
                "path": row["path"],
                "sha256": row["sha256"],
                "source_path": row["source_path"],
                "source_sha256": row["source_sha256"],
                "width": row["width"],
                "height": row["height"],
                "split": row["split"],
                "mask_path": row.get("mask_path"),
                "mask_sha256": row.get("mask_sha256"),
            } for row in fake],
        }
        bases.append(base)
        selected_inputs.extend(real + fake)

    # Hash-verify every selected q95, source and mask file against its manifest.
    verification = {"q95_checked": 0, "source_checked": 0, "mask_checked": 0,
                    "failures": []}
    for row in selected_inputs:
        for key, path_key, hash_key, counter in (
                ("q95", "path", "sha256", "q95_checked"),
                ("source", "source_path", "source_sha256", "source_checked")):
            path = ROOT / row[path_key]
            if not path.exists():
                verification["failures"].append(f"{key}: missing {path}")
                continue
            actual = sha256(path)
            verification[counter] += 1
            if actual != row[hash_key]:
                verification["failures"].append(
                    f"{key}: hash mismatch {row[path_key]} {actual} != {row[hash_key]}")
        mask_path = row.get("mask_path")
        if mask_path:
            path = ROOT / mask_path
            if not path.exists():
                verification["failures"].append(f"mask: missing {mask_path}")
            else:
                actual = sha256(path)
                verification["mask_checked"] += 1
                if actual != row.get("mask_sha256"):
                    verification["failures"].append(
                        f"mask: hash mismatch {mask_path}")
    if verification["failures"]:
        raise SystemExit("hash verification failed: " +
                         "; ".join(verification["failures"][:5]))

    counts = {
        "bases": len(bases),
        "inputs": len(selected_inputs),
        "real": sum(1 for row in selected_inputs if row["label"] == "Real"),
        "fake": sum(1 for row in selected_inputs if row["label"] == "Fake"),
        "by_role": dict(roles),
        "by_area_bin": dict(Counter(base["area_bin"] for base in bases
                                    if base["variant_count"] > 0)),
        "by_size_bin": dict(Counter(base["size_bin"] for base in bases
                                    if base["variant_count"] == 0)),
    }
    if counts["bases"] != 100 or counts["inputs"] != 220:
        raise SystemExit(f"unexpected subset size: {counts}")
    if counts["real"] != 100 or counts["fake"] != 120:
        raise SystemExit(f"unexpected label counts: {counts}")
    if counts["by_role"] != {"authentic_only": 30, "forged_single": 35,
                             "forged_double": 20, "forged_triple": 15}:
        raise SystemExit(f"unexpected role counts: {counts['by_role']}")

    payload = {
        "schema": "autosplice_localization_subset/1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "source_manifests": {
            "pairing": "dataset/AutoSplice_q95/pairing_manifest.json",
            "inputs": "dataset/AutoSplice_q95/inputs_manifest.json",
            "splits": "dataset/AutoSplice_q95/split_manifest.json",
        },
        "rules": {
            "partition": "dev only",
            "unit": "base/source; all variants of a base stay together",
            "mask": "shared by base; no per-variant mask exists",
            "area_stratification": "tertiles over dev forged bases",
            "size_stratification": "tertiles over dev authentic-only bases",
            "model_runs": 0,
        },
        "quotas": QUOTAS,
        "area_cuts": [round(float(value), 8) for value in area_cuts],
        "size_cuts": [round(float(value), 8) for value in size_cuts],
        "counts": counts,
        "verification": verification,
        "notes": notes,
        "bases": bases,
        "inputs": selected_inputs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    digest = sha256(OUT)
    md = [
        "# AutoSplice q95 定位开发子集（2026-11-02 固定）",
        "",
        f"种子 `{SEED}`；先写清单、后跑模型。清单：`{OUT.relative_to(ROOT)}`"
        f"（sha256 `{digest}`）。",
        "",
        "| 项目 | 数量 |",
        "| --- | ---: |",
        f"| base | {counts['bases']} |",
        f"| 输入 | {counts['inputs']} |",
        f"| Real | {counts['real']} |",
        f"| Fake | {counts['fake']} |",
        f"| authentic-only base | {counts['by_role']['authentic_only']} |",
        f"| 1 变体 base | {counts['by_role']['forged_single']} |",
        f"| 2 变体 base | {counts['by_role']['forged_double']} |",
        f"| 3 变体 base | {counts['by_role']['forged_triple']} |",
        "",
        f"mask 面积切点：{payload['area_cuts']}；authentic-only 尺寸切点："
        f"{payload['size_cuts']}。",
        "",
        "所有选中 base 的 authentic + 全部 forged 变体一起进入；mask 为 base 级共享，"
        "同源变体不能当独立源。q95、源图和 mask 的哈希已逐项与清单核对："
        f"q95 {verification['q95_checked']}、源图 {verification['source_checked']}、"
        f"mask {verification['mask_checked']}，失败 "
        f"{len(verification['failures'])}。",
        "",
        "本轮无 GPU/2 GiB cgroup，未跑 TruFor/IML-ViT；子集只做固定，不产出分数。",
    ]
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(OUT.relative_to(ROOT)), "sha256": digest,
                      "counts": counts, "verification": verification},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
