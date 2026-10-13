"""
The hard gate on the adopted conclusion (G7 §4.17 R4).

Four things that used to be one `pass`: the evidence check, the format of the
adopted answer, whether the conclusion is faithful, and whether the sample is
trainable.  A `failed` check must not reach training — not through admission
scoring, not through conversion — while a legacy trajectory that predates the
check is not retroactively judged (its audit is a separate, recorded exercise).
"""

import json

import pytest

from utils.final_v2 import conclusion_gate

PASS = {"status": "pass", "findings": [], "entry": "closing_turn"}
OK_CONCLUSION = {"source": "closing_turn", "calls": [], "calls_executed": False,
                 "verdict_parse": "ok"}


def _trajectory(**overrides):
    from tests.test_final_v2 import _record

    return _record(**overrides)


class TestTheGateItself:
    def test_a_legacy_record_is_not_retroactively_judged(self):
        assert conclusion_gate(_trajectory()) is None      # no evidence_check key

    def test_a_failed_check_is_refused(self):
        record = _trajectory(
            evidence_check={"status": "failed", "findings": [{"kind": "x"}],
                            "entry": "closing_turn"},
            conclusion=OK_CONCLUSION)
        assert conclusion_gate(record) == "evidence_check_failed"

    def test_a_session_that_never_ran_the_check_is_refused(self):
        record = _trajectory(evidence_check={"status": "not_run", "findings": []})
        assert conclusion_gate(record) == "evidence_check_not_run"

    def test_a_correction_that_passed_is_trainable(self):
        record = _trajectory(
            evidence_check={**PASS, "status": "pass_after_correction"},
            conclusion=OK_CONCLUSION)
        assert conclusion_gate(record) is None

    def test_an_unparseable_conclusion_is_refused(self):
        record = _trajectory(
            evidence_check=PASS,
            conclusion={**OK_CONCLUSION, "verdict_parse": "failed"})
        assert conclusion_gate(record) == "conclusion_without_parseable_verdict"

    def test_an_unexecuted_call_in_the_conclusion_is_refused(self):
        record = _trajectory(
            evidence_check=PASS,
            conclusion={**OK_CONCLUSION, "calls": ["noise"]})
        assert conclusion_gate(record) == "conclusion_contains_unexecuted_calls"

    def test_a_call_the_controller_executed_is_fine(self):
        record = _trajectory(
            evidence_check=PASS,
            conclusion={**OK_CONCLUSION, "calls": ["noise"], "calls_executed": True})
        assert conclusion_gate(record) is None

    def test_a_pre_r4_record_without_a_conclusion_block_still_passes(self):
        assert conclusion_gate(_trajectory(evidence_check=PASS)) is None


class TestAdmissionRefusesAGatedRecord:
    """`failed` is a hard gate, not a footnote beside good numbers."""

    APPLICABILITY = {"noise": {"png": 0.845}}

    def _admit(self, **overrides):
        from scripts.admit_trajectories_g4 import admit
        from tests.test_admit_trajectories_g4 import _record

        tool = _record("noise", ("noise",), brier=0.01)
        tool.update(overrides)
        baseline = _record("no-tool", (), brier=0.5)
        return admit(tool, baseline, self.APPLICABILITY)

    def test_the_same_record_is_positive_without_the_gate(self):
        decision = self._admit()
        assert decision["category"] == "positive"
        assert decision["evidence_gate"] is None

    def test_a_failed_check_forces_reject(self):
        decision = self._admit(
            evidence_check={"status": "failed", "findings": [{"kind": "x"}]},
            conclusion=OK_CONCLUSION)
        assert decision["category"] == "reject"
        assert decision["evidence_gate"] == "evidence_check_failed"
        assert any("evidence gate" in reason for reason in decision["reasons"])

    def test_the_report_counts_the_gate_rejections(self):
        from scripts.admit_trajectories_g4 import run
        from tests.test_admit_trajectories_g4 import _record

        gated = _record("noise", ("noise",), brier=0.01, variant="v1")
        gated.update(evidence_check={"status": "failed", "findings": [{"kind": "x"}]},
                     conclusion=OK_CONCLUSION)
        records = [_record("no-tool", (), brier=0.5, variant="v1"), gated]
        summary = run(records, self.APPLICABILITY)
        assert summary["evidence_gate"] == {"evidence_check_failed": 1}
        assert summary["evidence_gate_rejects"] == 1


class TestConversionRefusesAndKeepsTheOriginal:
    def _batch(self, tmp_path, record):
        trajectories = tmp_path / "trajectories"
        trajectories.mkdir()
        (trajectories / "t.json").write_text(json.dumps(record), encoding="utf-8")
        admission = tmp_path / "admission.json"
        admission.write_text(json.dumps({"decisions": [
            {"trajectory_id": record["trajectory_id"], "category": "positive",
             "policy": record["policy"], "reasons": []}]}), encoding="utf-8")
        return trajectories, admission, tmp_path / "out"

    def _build(self, tmp_path, record):
        from scripts.build_final_v2 import build

        trajectories, admission, out = self._batch(tmp_path, record)
        metadata = build(str(trajectories), str(admission), str(out), "g4_trial",
                         variants_manifest=str(tmp_path / "variants.json"))
        return metadata, out, trajectories

    def test_a_gated_record_is_refused_and_recorded(self, tmp_path):
        record = _trajectory(
            evidence_check={"status": "failed", "findings": [{"kind": "x"}]},
            conclusion=OK_CONCLUSION)
        metadata, out, trajectories = self._build(tmp_path, record)
        assert metadata["total"] == 0 and metadata["gated"] == 1
        assert metadata["gate_reasons"] == {"evidence_check_failed": 1}
        gated = json.loads((out / "sft_gated.json").read_text(encoding="utf-8"))
        assert gated == [{"trajectory_id": record["trajectory_id"],
                          "category": "positive", "reason": "evidence_check_failed"}]
        # the original trajectory is untouched: refusal is not deletion
        on_disk = json.loads((trajectories / "t.json").read_text(encoding="utf-8"))
        assert on_disk["evidence_check"]["status"] == "failed"

    def test_a_legacy_record_still_reaches_rendering(self, tmp_path):
        record = _trajectory()          # no evidence_check key at all
        metadata, out, _ = self._build(tmp_path, record)
        assert metadata["gated"] == 0
        assert metadata["legacy_without_evidence_check"] == 1
        assert json.loads((out / "sft_gated.json").read_text(encoding="utf-8")) == []


class TestTheSchemaIsEnforcedForNewBatches:
    """A missing field in a batch that declares the schema is a defect."""

    def test_a_declared_record_without_the_fields_is_refused(self):
        record = _trajectory(record_schema="g7_r4")
        record.pop("evidence_check", None)
        assert conclusion_gate(record) == "evidence_check_missing_for_schema"
        record = _trajectory(record_schema="g7_r4", evidence_check=PASS)
        assert conclusion_gate(record) == "conclusion_missing_for_schema"

    def test_a_declared_record_with_the_fields_passes(self):
        record = _trajectory(record_schema="g7_r4", evidence_check=PASS,
                             conclusion=OK_CONCLUSION)
        assert conclusion_gate(record) is None

    def test_an_undeclared_record_keeps_the_legacy_path(self):
        """History is not retroactively re-judged — but it is counted."""
        assert conclusion_gate(_trajectory()) is None
