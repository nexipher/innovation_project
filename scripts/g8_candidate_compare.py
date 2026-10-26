"""
Descriptive error overlap between PROBE and a candidate (plan §4.17c).

No hard gate: this script only counts, per group/condition and per source:

  probe wrong & candidate right   (correction)
  probe right & candidate wrong   (harm)
  both right
  both wrong

The independent unit is the **source**, not the image/format.  A source's
original/q95/q85/q70 measurements are repeated measures; the net-correction
CI is a source-clustered bootstrap (resample sources, carrying all their
conditions).  The 20 DIV2K Real sources are reused across the four generator
pairs; they are reported per pair and never counted as four independent sets.

Failures are reported separately; they are never treated as a wrong
prediction.  This is descriptive analysis for deciding what to keep later,
not an automatic admission rule.

  python scripts/g8_candidate_compare.py \
      --probe sft_data/g8_expert_eval/dev1_probe_... \
      --candidate sft_data/g8_expert_eval/dev1_spai_... \
      --out sft_data/g8_expert_eval/compare_probe_spai.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROJECT_ROOT  # noqa: E402


def _load_scores(path: str) -> Tuple[Dict[str, dict], dict]:
    if os.path.isdir(path):
        path = os.path.join(path, "scores.jsonl")
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    by_variant = {}
    experts = set()
    for row in rows:
        experts.add(row["expert"])
        if row["variant_id"] in by_variant:
            raise SystemExit(f"重复 variant_id：{row['variant_id']}（{path}）")
        by_variant[row["variant_id"]] = row
    if len(experts) != 1:
        raise SystemExit(f"期望一个专家，实际 {sorted(experts)}：{path}")
    return by_variant, {"path": os.path.relpath(path, PROJECT_ROOT),
                        "expert": sorted(experts)[0], "rows": len(rows)}


def _measured(row: dict) -> bool:
    return (row.get("status", "measured") == "measured"
            and row.get("value") is not None)


def _wrong(row: dict) -> bool:
    """Threshold decision from the row's registered direction and threshold."""
    value = float(row["value"])
    threshold = float(row.get("threshold", 0.5))
    direction = row.get("direction", "higher_is_fake")
    predicted_fake = (value >= threshold if direction == "higher_is_fake"
                      else value < threshold)
    return predicted_fake != (row["label"] == "Fake")


def _source(row: dict) -> str:
    return row.get("source_id") or row.get("source_path") or row["variant_id"]


def compare(probe: Dict[str, dict], candidate: Dict[str, dict]) -> dict:
    common = sorted(set(probe) & set(candidate))
    probe_only = sorted(set(probe) - set(candidate))
    candidate_only = sorted(set(candidate) - set(probe))
    cells = {"correction": [], "harm": [], "both_right": [], "both_wrong": []}
    failures = {"probe": [], "candidate": [], "either": []}
    per_group: Dict[str, dict] = defaultdict(
        lambda: {"cells": {name: [] for name in cells},
                 "sources": {name: set() for name in cells}})
    per_condition: Dict[str, dict] = defaultdict(
        lambda: {"cells": {name: 0 for name in cells},
                 "sources": {name: set() for name in cells}})
    source_cells: Dict[str, List[str]] = defaultdict(list)

    for variant in common:
        p, c = probe[variant], candidate[variant]
        # Both files must describe the same input; a mismatch here would make
        # the 2x2 silently compare different images.
        same_input = (p.get("label") == c.get("label")
                      and p.get("group") == c.get("group")
                      and p.get("condition") == c.get("condition")
                      and _source(p) == _source(c))
        if not same_input:
            raise SystemExit(
                f"同一 variant_id 的元数据不一致：{variant}｜"
                f"probe=({p.get('label')},{p.get('group')},{p.get('condition')},"
                f"{_source(p)})｜candidate=({c.get('label')},{c.get('group')},"
                f"{c.get('condition')},{_source(c)})")
        if not _measured(p):
            failures["probe"].append(variant)
        if not _measured(c):
            failures["candidate"].append(variant)
        if not (_measured(p) and _measured(c)):
            failures["either"].append(variant)
            continue
        p_wrong, c_wrong = _wrong(p), _wrong(c)
        if p_wrong and not c_wrong:
            cell = "correction"
        elif not p_wrong and c_wrong:
            cell = "harm"
        elif not p_wrong and not c_wrong:
            cell = "both_right"
        else:
            cell = "both_wrong"
        cells[cell].append(variant)
        source = _source(p)
        source_cells[source].append(cell)
        group = p["group"]
        per_group[group]["cells"][cell].append(variant)
        per_group[group]["sources"][cell].add(source)
        condition = p["condition"]
        per_condition[condition]["cells"][cell] += 1
        per_condition[condition]["sources"][cell].add(source)

    def summarize(cell_map, source_map):
        return {
            "cells": {name: len(values) for name, values in cell_map.items()},
            "sources": {name: len(values) for name, values in source_map.items()},
            "sample_ids": {name: sorted(values)
                           for name, values in cell_map.items()},
        }

    # Source-clustered bootstrap for net correction (descriptive only).
    sources = sorted(source_cells)
    rng = random.Random(20261026)
    nets = []
    for _ in range(1000):
        sample = [rng.choice(sources) for _ in sources] if sources else []
        corrections = sum(1 for s in sample for cell in source_cells[s]
                          if cell == "correction")
        harms = sum(1 for s in sample for cell in source_cells[s]
                    if cell == "harm")
        nets.append(corrections - harms)
    nets.sort()
    net = len(cells["correction"]) - len(cells["harm"])
    ci = ([nets[int(0.025 * (len(nets) - 1))],
           nets[int(0.975 * (len(nets) - 1))]] if nets else None)

    return {
        "schema": "candidate_error_overlap/1",
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "note": ("Descriptive only; no hard gate. Independent unit = source; "
                 "same-source formats are repeated measures. DIV2K real sources "
                 "are reused across generator pairs and are not four independent "
                 "sets. Rows missing from either file are counted separately and "
                 "never treated as a cell."),
        "common_variants": len(common),
        "overall": {
            "cells": {name: len(values) for name, values in cells.items()},
            "sources": {name: len({_source(probe[v]) for v in values})
                        for name, values in cells.items()},
            "sample_ids": {name: sorted(values) for name, values in cells.items()},
            "n_independent_sources": len({_source(probe[v]) for v in common}),
            "n_independent_sources_measured": len(sources),
            "n_measured_source_condition_cells": sum(
                len(values) for values in source_cells.values()),
            "net_correction": net,
            "net_correction_source_cluster_ci95": ci,
            "bootstrap": 1000,
            "failures": {name: len(values) for name, values in failures.items()},
            "failure_ids": {name: sorted(values)
                            for name, values in failures.items()},
            "missing": {
                "probe_only": len(probe_only),
                "candidate_only": len(candidate_only),
                "probe_only_ids": probe_only,
                "candidate_only_ids": candidate_only,
            },
        },
        "per_group": {group: summarize(data["cells"], data["sources"])
                      for group, data in sorted(per_group.items())},
        "per_condition": {
            condition: {
                "cells": dict(data["cells"]),
                "sources": {name: len(values)
                            for name, values in data["sources"].items()},
            }
            for condition, data in sorted(per_condition.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    probe, probe_meta = _load_scores(args.probe)
    candidate, candidate_meta = _load_scores(args.candidate)
    report = compare(probe, candidate)
    report["inputs"] = {"probe": probe_meta, "candidate": candidate_meta}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    overall = report["overall"]
    print(json.dumps({
        "probe": probe_meta["expert"], "candidate": candidate_meta["expert"],
        "common_variants": report["common_variants"],
        "cells": overall["cells"], "sources": overall["sources"],
        "n_independent_sources": overall["n_independent_sources"],
        "n_independent_sources_measured": overall["n_independent_sources_measured"],
        "net_correction": overall["net_correction"],
        "net_ci95": overall["net_correction_source_cluster_ci95"],
        "failures": overall["failures"],
        "missing": {name: overall["missing"][name]
                    for name in ("probe_only", "candidate_only")},
    }, ensure_ascii=False))
    print(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
