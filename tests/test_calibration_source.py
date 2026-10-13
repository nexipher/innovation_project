"""
Where the calibration numbers come from, and which one weights a token.

The A2 audit found two defects in the runtime table: the mapping was fitted on
A+B+C (392 samples) although the scheme says fitting reads group A only, and
the expert's weight was the *png cell's* separation applied to every format —
the JPEG expert carried 0.972 on an image whose own q70 cell measures 0.401.

Both are fixed in the builder; the previous table is kept for replay.
"""

import json
import os
import sys

import pytest

from config import PROJECT_ROOT
from utils.reliability import ReliabilityTable

TABLE = os.path.join(PROJECT_ROOT, "calibration", "reliability_table.json")
TABLE_V1 = os.path.join(PROJECT_ROOT, "calibration", "reliability_table_v1.json")
SPLIT = os.path.join(PROJECT_ROOT, "calibration", "set", "g7_split.json")
G2_EXPERTS = ("noise_expert", "jpeg_expert", "frequency_expert_v2", "ela_expert")


def _table():
    with open(TABLE, encoding="utf-8") as handle:
        return json.load(handle)


class TestTheTableDeclaresItsSource:
    def test_it_is_fitted_on_the_designated_group(self):
        table = _table()
        assert table["schema"] == 2
        assert table["fitting_group"] == "A"
        split_hash = json.load(open(SPLIT, encoding="utf-8"))["hash"]
        assert table["fitting_split_hash"] == split_hash

    def test_every_expert_carries_its_provenance(self):
        for name, entry in _table()["experts"].items():
            provenance = entry.get("provenance") or {}
            assert provenance.get("fitted_on") == "A", name
            assert provenance.get("split_hash"), name

    def test_the_g2_mappings_are_no_longer_fitted_on_b_or_c(self):
        """392 was A+B+C; the fitting group holds 248 matched samples."""
        table = _table()
        group_a = len(json.load(open(SPLIT, encoding="utf-8"))["groups"]["A"])
        for name in G2_EXPERTS:
            entry = table["experts"][name]
            assert entry["n_samples"] == group_a * 4, name       # 62 sources x 4 matched cells
            assert entry["n_samples"] != 392, name


class TestTheWeightFollowsTheFormat:
    def test_every_runtime_expert_has_per_format_separations(self):
        table = _table()
        for name in G2_EXPERTS + ("probe_dino_expert",):
            per_cell = table["experts"][name].get("per_cell_separation") or {}
            assert set(per_cell) >= {"png", "jpeg_q95", "jpeg_q85", "jpeg_q70"}, name

    def test_a_known_format_uses_its_own_cell(self):
        entry = _table()["experts"]["jpeg_expert"]
        weight, basis = ReliabilityTable.weight_of(entry, "jpeg_q70")
        assert basis == "condition:jpeg_q70"
        assert weight == entry["per_cell_separation"]["jpeg_q70"]
        assert weight != entry["per_cell_separation"]["png"]

    def test_an_unknown_format_falls_back_to_the_pooled_value(self):
        entry = _table()["experts"]["jpeg_expert"]
        weight, basis = ReliabilityTable.weight_of(entry, None)
        assert basis == "pooled"
        assert weight == entry["separation_polarity_corrected"]

    def test_the_lookup_reports_which_measurement_it_used(self):
        table = ReliabilityTable.load()
        full = table.lookup("jpeg_expert", 3.5, condition="jpeg_q85")
        assert full["condition_metadata"]["weight_basis"] == "condition:jpeg_q85"
        assert full["condition_metadata"]["condition"] == "jpeg_q85"
        assert full["reliability"] == pytest.approx(0.7219, abs=1e-3)
        pooled = table.lookup("jpeg_expert", 3.5)
        assert pooled["condition_metadata"]["weight_basis"] == "pooled"
        assert pooled["reliability"] == pytest.approx(0.6623, abs=1e-3)

    def test_the_png_cell_is_not_used_for_a_recompressed_image(self):
        table = ReliabilityTable.load()
        png = table.lookup("jpeg_expert", 3.5, condition="png")["reliability"]
        q70 = table.lookup("jpeg_expert", 3.5, condition="jpeg_q70")["reliability"]
        assert png - q70 > 0.4, "A2 的缺陷正是把 png 的分离度用到了 q70 上"


class TestTheOldTableSurvives:
    def test_v1_is_kept_for_replay(self):
        with open(TABLE_V1, encoding="utf-8") as handle:
            old = json.load(handle)
        assert "schema" not in old                       # the pre-repair table
        entry = old["experts"]["jpeg_expert"]
        assert entry["n_samples"] == 392                 # A+B+C, as it was fitted
        # v1's weight came from the png cell and was applied to every format
        with open(TABLE, encoding="utf-8") as handle:
            new = json.load(handle)["experts"]["jpeg_expert"]
        assert entry["separation_polarity_corrected"] == 0.9720833333333333
        assert abs(new["separation_polarity_corrected"] - 0.9721) > 0.2


class TestTheBuilderRefusesToGuessTheGroup:
    def test_a_missing_split_stops_the_build(self, monkeypatch, tmp_path):
        import scripts.build_reliability_table as builder

        monkeypatch.setattr(builder, "SPLIT_PATH", str(tmp_path / "missing.json"))
        monkeypatch.setattr(builder, "TABLE_PATH", str(tmp_path / "out.json"))
        monkeypatch.setattr(sys, "argv", ["build_reliability_table.py"])
        with pytest.raises(SystemExit, match="fitting split is required"):
            builder.main()

    def test_a_rebuild_is_reproducible(self, monkeypatch, tmp_path):
        import scripts.build_reliability_table as builder

        out = tmp_path / "table.json"
        monkeypatch.setattr(builder, "TABLE_PATH", str(out))
        monkeypatch.setattr(sys, "argv", ["build_reliability_table.py"])
        builder.main()
        with open(out, encoding="utf-8") as handle:
            rebuilt = json.load(handle)
        assert rebuilt["fitting_group"] == "A"
        assert (rebuilt["experts"]["noise_expert"]["per_cell_separation"]
                == _table()["experts"]["noise_expert"]["per_cell_separation"])
        assert (rebuilt["experts"]["noise_expert"]["bins"]
                == _table()["experts"]["noise_expert"]["bins"])


class TestTheRunKnowsWhichTableItRead:
    def test_the_fingerprint_carries_the_calibration_identity(self):
        from utils.config_fingerprint import compute, calibration_identity

        identity = calibration_identity()
        assert identity["schema"] == 2 and identity["fitting_group"] == "A"
        live = compute({})
        assert live["calibration"] == identity
        assert live["halting_policy"]

    def test_a_missing_table_is_reported_as_missing(self, tmp_path):
        from utils.config_fingerprint import calibration_identity

        identity = calibration_identity(str(tmp_path / "none.json"))
        assert identity == {"digest": None, "schema": None, "fitting_group": None}


class TestTheSessionRecordsItsCondition:
    def test_a_session_records_the_cell_it_weighted_for(self, tmp_path):
        from state_machine.controller import ForensicStateMachine
        from tests.test_controller import FAKE_PATH, ScriptedMLLM, _build_experts

        from utils.logger import SessionLogger

        verdict = ('<reasoning>噪声残留专家给出测量。</reasoning>\n<verdict>'
                   '{"verdict": "Uncertain", "confidence": 0.4, "primary_evidence": [],'
                   ' "report": "证据有限。"}</verdict>')
        fsm = ForensicStateMachine(ScriptedMLLM([verdict]), _build_experts(),
                                   logger=SessionLogger(sft_dir=str(tmp_path)),
                                   condition="jpeg_q70")
        result = fsm.run(FAKE_PATH, "Fake")
        with open(result["sft_data_path"], encoding="utf-8") as handle:
            assert json.load(handle)["metadata"]["calibration_condition"] == "jpeg_q70"
