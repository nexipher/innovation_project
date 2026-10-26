"""Descriptive PROBE vs candidate error overlap (plan §4.17c)."""

import pytest

from scripts.g8_candidate_compare import compare


def _row(variant, group, condition, label, value, expert,
         direction="higher_is_fake", status="measured"):
    return {"variant_id": variant, "expert": expert, "group": group,
            "condition": condition, "label": label,
            "source_id": variant.split("__")[0],
            "source_path": variant.split("__")[0],
            "value": value, "oriented": (value if value is not None else None),
            "direction": direction, "threshold": 0.5,
            "status": status}


def _pair(variant, group, condition, label, probe_value, candidate_value,
          probe_status="measured"):
    return (_row(variant, group, condition, label, probe_value, "probe",
                 status=probe_status),
            _row(variant, group, condition, label, candidate_value, "spai"))


def _data():
    pairs = [
        _pair("S1__a__original", "gen", "original", "Fake", 0.4, 0.9),   # correction
        _pair("S1__a__jpeg_q95", "gen", "jpeg_q95", "Fake", 0.9, 0.9),   # both right
        _pair("S2__a__original", "gen", "original", "Fake", 0.9, 0.4),   # harm
        _pair("S3__a__original", "gen2", "original", "Fake", 0.9, 0.9),  # both right
        _pair("S4__a__original", "gen2", "original", "Fake", 0.4, 0.4),  # both wrong
        _pair("S5__a__original", "gen", "original", "Fake", None, 0.9,
              probe_status="failed"),                           # failure, not a cell
    ]
    probe = {p["variant_id"]: p for p, _c in pairs}
    candidate = {c["variant_id"]: c for _p, c in pairs}
    return probe, candidate


class TestErrorOverlap:
    def test_cells_and_sources_are_counted_by_source(self):
        probe, candidate = _data()
        report = compare(probe, candidate)
        assert report["common_variants"] == 6
        assert report["overall"]["cells"] == {
            "correction": 1, "harm": 1, "both_right": 2, "both_wrong": 1}
        assert report["overall"]["sources"] == {
            "correction": 1, "harm": 1, "both_right": 2, "both_wrong": 1}
        assert report["overall"]["net_correction"] == 0

    def test_a_failure_is_not_a_wrong_prediction(self):
        probe, candidate = _data()
        report = compare(probe, candidate)
        assert report["overall"]["failures"]["probe"] == 1
        assert "S5__a__original" in report["overall"]["failure_ids"]["probe"]
        assert sum(report["overall"]["cells"].values()) == 5
        assert report["overall"]["n_independent_sources"] == 5
        assert report["overall"]["n_independent_sources_measured"] == 4
        assert report["overall"]["n_measured_source_condition_cells"] == 5

    def test_a_metadata_mismatch_is_rejected(self):
        probe, candidate = _data()
        candidate["S1__a__original"] = {
            **candidate["S1__a__original"], "group": "other-generator"}
        with pytest.raises(SystemExit, match="元数据不一致"):
            compare(probe, candidate)

    def test_rows_missing_from_one_file_are_reported(self):
        probe, candidate = _data()
        missing = candidate.pop("S1__a__jpeg_q95")
        report = compare(probe, candidate)
        assert report["overall"]["missing"]["candidate_only"] == 0
        assert report["overall"]["missing"]["probe_only"] == 1
        assert report["overall"]["missing"]["probe_only_ids"] == [
            "S1__a__jpeg_q95"]
        assert report["common_variants"] == 5
        assert missing["variant_id"] == "S1__a__jpeg_q95"

    def test_the_source_cluster_bootstrap_is_deterministic(self):
        probe, candidate = _data()
        first = compare(probe, candidate)["overall"][
            "net_correction_source_cluster_ci95"]
        second = compare(probe, candidate)["overall"][
            "net_correction_source_cluster_ci95"]
        assert first == second

    def test_it_has_no_hard_gate(self):
        probe, candidate = _data()
        report = compare(probe, candidate)
        assert "gate" not in report
        assert "descriptive only" in report["note"].lower()
