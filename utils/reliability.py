"""
Empirical reliability lookup for forensic experts (G2 §4.9).

The G2 calibration distilled `calibration/reliability_table.json` from
per-sample measurements: for each expert it records the polarity-corrected
separation, the format-confound drop, applicability notes and an empirical
binning of the raw metric -> P(Fake). The runtime pipeline consults this
table so evidence tokens carry calibrated numbers instead of raw scores.

Degrades gracefully: when the table is missing, lookup returns None and the
pipeline behaves exactly as before G2.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from config import PROJECT_ROOT

RELIABILITY_TABLE_PATH = os.path.join(
    PROJECT_ROOT, "calibration", "reliability_table.json"
)


class ReliabilityTable:
    """Loader and lookup API for the distilled calibration table."""

    def __init__(self, data: Dict[str, Any]):
        self._data = data
        self._experts = data.get("experts", {})

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str = RELIABILITY_TABLE_PATH) -> Optional["ReliabilityTable"]:
        """Load the table; return None when unavailable or unreadable."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, encoding="utf-8") as handle:
                return cls(json.load(handle))
        except (json.JSONDecodeError, OSError):
            return None

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def expert_entry(self, source: str) -> Optional[dict]:
        return self._experts.get(source)

    def lookup(self, source: str, raw_metric: float) -> Optional[dict]:
        """
        Look up calibration for one measurement.

        Args:
            source: Expert source name, e.g. "noise_expert".
            raw_metric: The expert's raw metric for this analysis.

        Returns:
            dict with keys `reliability`, `calibrated_likelihood`,
            `semantics_aligned`, `applicability`, `condition_metadata`
            (None when the expert is unknown to the table).
        """
        entry = self._experts.get(source)
        if not entry:
            return None

        probability_fake = None
        bin_note = None
        for bucket in entry.get("bins", []):
            # None bounds mark open-ended outer bins: (-inf, hi) / (lo, +inf).
            low = bucket.get("lo")
            high = bucket.get("hi")
            if low is not None and raw_metric < low:
                continue
            if high is not None and raw_metric >= high:
                continue
            probability_fake = bucket.get("p_fake")
            bin_note = bucket.get("n")
            break

        if probability_fake is None:
            return None

        likelihood = {
            "Real": 1.0 - probability_fake,
            "Fake": probability_fake,
        }
        return {
            "reliability": float(entry.get("separation_polarity_corrected", 0.5)),
            "calibrated_likelihood": likelihood,
            "semantics_aligned": bool(entry.get("semantics_aligned", True)),
            "applicability": entry.get("applicability", "unknown"),
            "applicability_conditions": entry.get("applicability_conditions", ""),
            "condition_metadata": {
                "bin_samples": bin_note,
                "confound_drop_png_to_q70": entry.get("confound_drop_png_to_q70"),
                "separation_polarity_corrected": entry.get("separation_polarity_corrected"),
            },
        }
