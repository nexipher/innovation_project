"""Add two supplementary single-expert exemplars to the 24-sample manifest.

The 24 original rows keep their source/asset identity.  Two new rows reuse the
same images:
  * `single_probe__...Glide...`  — PROBE suffices, conclude directly;
  * `single_trufor__...54029...` — TruFor suffices for overall Fake, whole
    image stays Uncertain.

The original manifest/scripts are copied to `*_round1.json` once, then the
working `manifest.json` is replaced with 26 rows in review order:
8 key cases, 2 single-expert exemplars, 16 remaining.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
DEFAULT_OUT = ROOT / "sft_data/sft_candidates_20261127"

SINGLES = (
    ("genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95",
     "single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95",
     "single_expert_probe", "probe_only",
     ["single_expert", "probe_only", "whole_generated"]),
    ("autosplice__54029__1__q95",
     "single_trufor__autosplice__54029__1__q95",
     "single_expert_trufor", "trufor_only",
     ["single_expert", "trufor_only", "strong_local_evidence"]),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    manifest_path = out / "manifest.json"
    scripts_path = out / "candidate_scripts.json"
    round1_manifest = out / "manifest_round1.json"
    round1_scripts = out / "candidate_scripts_round1.json"
    if not round1_manifest.exists():
        shutil.copy2(manifest_path, round1_manifest)
    if not round1_scripts.exists():
        shutil.copy2(scripts_path, round1_scripts)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    old = manifest["samples"]
    by_id = {row["sample_id"]: row for row in old}
    singles = []
    for source_id, new_id, kind, role, tags in SINGLES:
        if source_id not in by_id:
            raise SystemExit(f"source sample missing: {source_id}")
        row = json.loads(json.dumps(by_id[source_id]))
        row["sample_id"] = new_id
        row["variant_id"] = new_id
        row["kind"] = kind
        row["selection_role"] = role
        row["risk_tags"] = tags
        row["key_case"] = False
        row["key_case_rank"] = None
        row["single_expert"] = True
        row["usage_flags"] = ["sft_candidate_round1", role]
        row["usage_roles"] = ["sft_candidate_review"]
        row["note"] = ("supplementary single-expert exemplar; same image and "
                       "source identity as " + source_id)
        singles.append(row)

    key = [row for row in old if row.get("key_case")]
    rest = [row for row in old if not row.get("key_case")]
    samples = key + singles + rest
    counts = {}
    for row in samples:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    manifest["samples"] = samples
    manifest["counts"] = {
        "total": len(samples),
        "by_kind": counts,
        "key_cases": len(key),
        "single_expert_exemplars": len(singles),
    }
    manifest["rules"]["composition"]["total"] = len(samples)
    manifest["rules"]["composition"]["single_expert"] = {
        "probe_only": 1, "trufor_only": 1}
    manifest["supplementary"] = {
        "purpose": ("single-expert exemplars: one PROBE-only conclusion and one "
                    "TruFor-only overall Fake with whole Uncertain"),
        "samples": [row["sample_id"] for row in singles],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "total": len(samples),
        "key_cases": len(key),
        "single_expert": [row["sample_id"] for row in singles],
        "order": [row["sample_id"] for row in samples[:11]],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
