"""Build the 200-input g8 dev evaluation manifest (2026-11-10).

Fixed composition:
  * 60 Real: 30 GenImage Real + 30 AutoSplice Authentic
  * 80 whole-generated: 10 from each GenImage generator
  * 60 local tampering: 60 distinct AutoSplice forged sources

Only existing dev/val inputs are used.  The 24 previously reused inputs and
their AutoSplice base variants are excluded.  The sample is deterministic from
SEED; no quota is reduced silently: shortages are reported and the script exits
without writing a manifest.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
GEN_MANIFESTS = (
    ROOT / "dataset/GenImage_q95/eval_dev_manifest_20261102.json",
    ROOT / "dataset/GenImage_q95/eval_val_manifest_20261102.json",
)
AUTOSPLICE_SUBSET = ROOT / "sft_data/g8_localization/autosplice_dev_subset_20261102.json"
PREVIOUS_MANIFEST = ROOT / "sft_data/g8_acceptance/dual_20261107/manifest.json"
DEFAULT_OUT = ROOT / "sft_data/g8_dev_eval_20261110/manifest.json"
SEED = 20261110
GROUPS = ("ADM", "BigGAN", "Glide", "Midjourney", "SD14", "SD15", "VQDM", "Wukong")
USED_AUTOSPLICE_BASES = {"39890", "41187", "72399", "42005", "57501",
                         "58610", "51136", "42165", "53059"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def rng(label: str) -> random.Random:
    # Per-quota RNG: changing one group's candidate pool does not perturb the
    # others.  The integer seed and this string are recorded in the manifest.
    return random.Random(f"{SEED}:{label}")


def previous_used():
    payload = load(PREVIOUS_MANIFEST)
    gen_sources = set()
    auto_bases = set()
    for row in payload["samples"]:
        if row["family"] == "GenImage_q95":
            gen_sources.add(row["source_id"])
        else:
            auto_bases.add(str(row["source_id"]))
    return payload, gen_sources, auto_bases


def verify_file(path: str, expected: str, kind: str, sample_id: str) -> str:
    actual_path = ROOT / path
    if not actual_path.is_file():
        raise SystemExit(f"{sample_id}: missing {kind}: {actual_path}")
    actual = sha256_file(actual_path)
    if expected and actual != expected:
        raise SystemExit(
            f"{sample_id}: {kind} sha256 mismatch: manifest={expected} actual={actual}")
    return actual


def gen_rows(used_sources: set[str]):
    rows = []
    for manifest_path in GEN_MANIFESTS:
        payload = load(manifest_path)
        split = payload["split"]
        if split not in ("dev", "val"):
            raise SystemExit(f"unexpected GenImage split {split!r} in {manifest_path}")
        for row in payload["inputs"]:
            if not row.get("no_known_use"):
                continue
            if row["source_id"] in used_sources:
                continue
            rows.append(dict(row, _pool_split=split, _manifest=str(manifest_path)))
    return rows


def pick_genimage(rows, group: str, n: int):
    candidates = sorted(
        (row for row in rows if row["group"] == group),
        key=lambda row: (row["source_id"], row["_pool_split"]))
    if len(candidates) < n:
        raise SystemExit(f"GenImage {group}: need {n}, only {len(candidates)} candidates")
    chosen = rng(f"genimage:{group}").sample(candidates, n)
    return sorted(chosen, key=lambda row: (row["_pool_split"], row["source_id"]))


def autosplice_bases():
    payload = load(AUTOSPLICE_SUBSET)
    bases = []
    for base in payload["bases"]:
        sid = str(base["source_id"])
        if sid in USED_AUTOSPLICE_BASES:
            continue
        bases.append(base)
    return payload, sorted(bases, key=lambda base: str(base["source_id"]))


def sample_autosplice(bases):
    forged_candidates = [b for b in bases if b.get("forged")]
    authentic_candidates = [b for b in bases if not b.get("forged")]
    if len(forged_candidates) < 60:
        raise SystemExit(
            f"AutoSplice forged: need 60 bases, only {len(forged_candidates)} after exclusions")
    forged_bases = rng("autosplice:forged").sample(forged_candidates, 60)
    forged_bases = sorted(forged_bases, key=lambda b: str(b["source_id"]))
    forged_ids = {str(b["source_id"]) for b in forged_bases}

    # Real and forged must not share a base.  Use every remaining base as a
    # candidate, not only authentic-only bases: after 60 forged bases, there are
    # 31 candidates left (26 authentic-only + forged bases not selected).
    authentic_pool = [b for b in bases if str(b["source_id"]) not in forged_ids]
    if len(authentic_pool) < 30:
        raise SystemExit(
            f"AutoSplice authentic: need 30 bases, only {len(authentic_pool)} after forged selection")
    authentic_bases = rng("autosplice:authentic").sample(authentic_pool, 30)
    authentic_bases = sorted(authentic_bases, key=lambda b: str(b["source_id"]))
    return forged_bases, authentic_bases


def genimage_row(raw, index: int):
    group = raw["group"]
    source_id = raw["source_id"]
    is_real = group == "Real"
    truth = ({"whole_image_generation": "Real",
              "local_manipulation": "Not tampered"} if is_real else
             {"whole_image_generation": "AI-generated",
              "local_manipulation": "Not applicable"})
    q95 = verify_file(raw["path"], raw["sha256"], "q95 image", source_id)
    source = verify_file(raw["source_path"], raw["source_sha256"], "source image", source_id)
    return {
        "sample_id": f"genimage__{source_id.replace('/', '__')}__q95",
        "variant_id": raw["variant_id"],
        "source_id": source_id,
        "family": "GenImage_q95",
        "group": group,
        "source_role": "genimage_real" if is_real else "genimage_generated",
        "path": raw["path"],
        "source_path": raw["source_path"],
        "q95_sha256": q95,
        "source_sha256": source,
        "mask_path": None,
        "mask_sha256": None,
        "mask_shared_by_base": False,
        "split": raw["new_split"],
        "old_split": raw.get("old_split"),
        "no_known_use": bool(raw.get("no_known_use")),
        "usage_flags": list(raw.get("usage_flags") or []),
        "usage_roles": list(raw.get("usage_roles") or []),
        "condition": raw.get("condition", "jpeg_q95"),
        "bytes": raw.get("bytes"),
        "width": raw.get("width"),
        "height": raw.get("height"),
        "ground_truth": "Real" if is_real else "Fake",
        "task_truth": truth,
        "risk_tags": ([] if is_real else ["whole_image_generated"]),
        "selection_index": index,
    }


def autosplice_row(base: dict, forged: bool, index: int):
    sid = str(base["source_id"])
    if forged:
        variants = sorted(base["forged"], key=lambda row: int(row["variant"]))
        chosen = rng(f"autosplice:variant:{sid}").choice(variants)
        path = chosen["path"]
        q95 = verify_file(path, chosen["sha256"], "AutoSplice q95 forged image", sid)
        source = verify_file(chosen["source_path"], chosen["source_sha256"],
                             "AutoSplice source forged image", sid)
        mask = verify_file(base["mask"]["path"], base["mask"]["sha256"],
                           "AutoSplice shared mask", sid)
        truth = {"whole_image_generation": "Real",
                 "local_manipulation": "Tampered"}
        role = "autosplice_forged"
        sample_id = f"autosplice__{sid}__{chosen['variant']}__q95"
        risk = ["local_edit"]
        if base.get("mask_shared_by_base"):
            risk.append("shared_mask_by_base")
        if len(variants) > 1:
            risk.append("shared_base_multiple_variants")
        extra = {
            "variant": str(chosen["variant"]),
            "base_variant_count": len(variants),
            "mask_shared_by_base": bool(base.get("mask_shared_by_base")),
            "area_bin": base.get("area_bin"),
            "selection_role": base.get("selection_role"),
        }
    else:
        auth = base["authentic"]
        path = auth["path"]
        q95 = verify_file(path, auth["sha256"], "AutoSplice q95 authentic image", sid)
        source = verify_file(auth["source_path"], auth["source_sha256"],
                             "AutoSplice source authentic image", sid)
        mask = None
        truth = {"whole_image_generation": "Real",
                 "local_manipulation": "Not tampered"}
        role = "autosplice_authentic"
        sample_id = f"autosplice__{sid}__authentic__q95"
        risk = ["autosplice_authentic"]
        if base.get("forged"):
            risk.append("authentic_from_forged_base")
        extra = {
            "base_variant_count": len(base.get("forged") or []),
            "mask_shared_by_base": False,
            "area_bin": None,
            "selection_role": base.get("selection_role"),
            "authentic_from_forged_base": bool(base.get("forged")),
        }
    return {
        "sample_id": sample_id,
        "variant_id": chosen.get("variant_id") if forged else auth["variant_id"],
        "source_id": sid,
        "family": "AutoSplice_q95",
        "group": "AutoSplice_Forged" if forged else "AutoSplice_Authentic",
        "source_role": role,
        "path": path,
        "source_path": chosen["source_path"] if forged else auth["source_path"],
        "q95_sha256": q95,
        "source_sha256": source,
        "mask_path": base["mask"]["path"] if forged else None,
        "mask_sha256": mask,
        "mask_shared_by_base": bool(base.get("mask_shared_by_base")) if forged else False,
        "split": "dev",
        "old_split": "not_in_old",
        "no_known_use": True,
        "usage_flags": [],
        "usage_roles": [],
        "condition": "jpeg_q95",
        "bytes": chosen.get("bytes") if forged else auth.get("bytes"),
        "width": chosen.get("width") if forged else auth.get("width"),
        "height": chosen.get("height") if forged else auth.get("height"),
        "encoder": chosen.get("encoder") if forged else None,
        "ground_truth": "Real" if not forged else "Fake",
        "task_truth": truth,
        "risk_tags": risk,
        "selection_index": index,
        **extra,
    }


def validate(samples: list[dict], previous_rows: list[dict], prev_bases: set[str]):
    errors = []
    ids = [row["sample_id"] for row in samples]
    variant_ids = [row["variant_id"] for row in samples]
    paths = [row["path"] for row in samples]
    q95s = [row["q95_sha256"] for row in samples]
    sources = [row["source_sha256"] for row in samples]
    for label, values in (("sample_id", ids), ("variant_id", variant_ids),
                          ("path", paths), ("q95_sha256", q95s),
                          ("source_sha256", sources)):
        duplicates = [value for value, count in collections.Counter(values).items()
                      if count > 1]
        if duplicates:
            errors.append(f"duplicate {label}: {duplicates[:3]}")
    # One input per source; AutoSplice real and forged bases are disjoint.
    auto_sources = [row["source_id"] for row in samples
                    if row["family"] == "AutoSplice_q95"]
    if len(auto_sources) != len(set(auto_sources)):
        errors.append("AutoSplice source appears in both real and forged")
    if set(auto_sources) & prev_bases:
        errors.append(f"previous AutoSplice base reused: {sorted(set(auto_sources) & prev_bases)}")
    gen_sources = [row["source_id"] for row in samples if row["family"] == "GenImage_q95"]
    if len(gen_sources) != len(set(gen_sources)):
        errors.append("GenImage source repeated")
    previous_q95 = {row["q95_sha256"] for row in previous_rows}
    if set(q95s) & previous_q95:
        errors.append("previous q95 hash reused")
    counts = collections.Counter(
        (row["family"], row["group"], row["task_truth"]["whole_image_generation"],
         row["task_truth"]["local_manipulation"]) for row in samples)
    expected = {
        ("GenImage_q95", "Real", "Real", "Not tampered"): 30,
        ("AutoSplice_q95", "AutoSplice_Authentic", "Real", "Not tampered"): 30,
        ("AutoSplice_q95", "AutoSplice_Forged", "Real", "Tampered"): 60,
    }
    for group in GROUPS:
        expected[("GenImage_q95", group, "AI-generated", "Not applicable")] = 10
    if dict(counts) != expected:
        errors.append(f"composition mismatch: {dict(counts)}")
    if len(samples) != 200:
        errors.append(f"expected 200 samples, got {len(samples)}")
    if errors:
        raise SystemExit("manifest validation failed:\n  " + "\n  ".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    previous, used_sources, used_bases = previous_used()
    all_gen = gen_rows(used_sources)
    samples = []
    index = 0
    for row in pick_genimage(all_gen, "Real", 30):
        samples.append(genimage_row(row, index)); index += 1
    for group in GROUPS:
        for row in pick_genimage(all_gen, group, 10):
            samples.append(genimage_row(row, index)); index += 1
    auto_payload, auto_bases = autosplice_bases()
    forged_bases, authentic_bases = sample_autosplice(auto_bases)
    for base in forged_bases:
        samples.append(autosplice_row(base, True, index)); index += 1
    for base in authentic_bases:
        samples.append(autosplice_row(base, False, index)); index += 1

    validate(samples, previous["samples"], used_bases)
    counts = collections.Counter()
    for row in samples:
        counts[(row["family"], row["group"],
                row["task_truth"]["whole_image_generation"],
                row["task_truth"]["local_manipulation"])] += 1
    payload = {
        "schema": "dev_eval_manifest/1",
        "created_at": "2026-11-10",
        "purpose": "200-session development evaluation on new dev/val sources; not training data",
        "seed": SEED,
        "rng": "random.Random(f'{seed}:{quota}') after sorting candidates by source_id",
        "rules": {
            "composition": {"total": 200, "real": 60, "whole_generated": 80,
                            "local_tampered": 60},
            "real": {"GenImage Real": 30, "AutoSplice Authentic": 30},
            "whole_generated": {group: 10 for group in GROUPS},
            "local_tampered": "60 distinct AutoSplice forged bases",
            "inputs": "one q95 input per source; AutoSplice real/forged bases disjoint",
            "splits": "GenImage dev+val; AutoSplice dev subset; test never read",
            "history": "GenImage no_known_use=True; previous 24 sources/bases excluded",
            "mask": "AutoSplice mask shared by base; no per-variant mask",
        },
        "pool": {
            "genimage": {"manifests": [str(path.relative_to(ROOT)) for path in GEN_MANIFESTS],
                          "candidates_after_exclusion": dict(collections.Counter(
                              row["group"] for row in all_gen))},
            "autosplice": {
                "subset": str(AUTOSPLICE_SUBSET.relative_to(ROOT)),
                "bases_total": len(auto_payload["bases"]),
                "bases_after_previous_exclusion": len(auto_bases),
                "forged_candidates": sum(1 for base in auto_bases if base.get("forged")),
                "authentic_candidates": sum(1 for base in auto_bases if not base.get("forged")),
                "bases_remaining_after_forged": len(auto_bases) - 60,
            },
        },
        "exclusions": {
            "previous_manifest": str(PREVIOUS_MANIFEST.relative_to(ROOT)),
            "previous_sample_count": len(previous["samples"]),
            "genimage_source_ids": sorted(used_sources),
            "autosplice_bases": sorted(used_bases),
            "q95_hashes": sorted({row["q95_sha256"] for row in previous["samples"]}),
        },
        "counts": {
            "total": len(samples),
            "by_family": dict(collections.Counter(row["family"] for row in samples)),
            "by_group": dict(collections.Counter(row["group"] for row in samples)),
            "by_truth": {
                "whole_image_generation": dict(collections.Counter(
                    row["task_truth"]["whole_image_generation"] for row in samples)),
                "local_manipulation": dict(collections.Counter(
                    row["task_truth"]["local_manipulation"] for row in samples)),
            },
        },
        "verification": {
            "samples_checked": len(samples),
            "q95_hashes_checked": len(samples),
            "source_hashes_checked": len(samples),
            "mask_hashes_checked": sum(1 for row in samples if row.get("mask_path")),
            "failures": [],
        },
        "samples": samples,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "counts": payload["counts"],
                      "pool": payload["pool"], "verification": payload["verification"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
