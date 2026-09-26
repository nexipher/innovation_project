"""Tests for G4-c two-level trajectory admission."""

import json

import pytest

from scripts.admit_trajectories_g4 import (
    CONFIDENT,
    RISK_EPSILON,
    admit,
    assert_train_only,
    load_trajectories,
    pair_by_variant,
    run,
)

APPLICABILITY = {
    "jpeg": {"png": 0.972, "jpeg_q70": 0.431},
    "noise": {"png": 0.845, "jpeg_q70": 0.772},
    "frequency_v2": {"png": 0.556, "jpeg_q70": 0.634},
}


def _record(policy, tools=(), treatment="png", verdict="Fake", confidence=0.9,
            ground_truth="Fake", brier=0.1, nll=0.2, cost=1.0, split="train",
            variant="v1", error=0.1):
    return {
        "trajectory_id": f"{policy}__{variant}",
        "policy": policy,
        "tools_served": list(tools),
        "variant_id": variant,
        "source_id": "ADM/x",
        "treatment": treatment,
        "generator": "ADM",
        "split": split,
        "ground_truth": ground_truth,
        "final_verdict": verdict,
        "confidence": confidence,
        "counters": {"weighted_cost": cost, "expert_calls": len(tools),
                     "model_turns": 2},
        "scores": {"posterior_brier": brier, "posterior_nll": nll,
                   "posterior_error": error},
    }


class TestTrainOnlyGuard:
    def test_train_records_pass(self):
        assert_train_only([_record("noise", ("noise",))])

    def test_any_other_partition_is_refused(self):
        with pytest.raises(ValueError, match="training partition"):
            assert_train_only([_record("noise", ("noise",), split="test")])

    def test_the_guard_sits_in_front_of_the_decisions(self):
        records = [_record("noise", ("noise",)), _record("noise", ("noise",), split="val")]
        with pytest.raises(ValueError):
            run(records, APPLICABILITY)


class TestConditionalLayer:
    def test_a_separating_tool_passes(self):
        decision = admit(_record("noise", ("noise",)), None, APPLICABILITY)
        assert decision["conditional_ok"] is True

    def test_a_tool_below_the_bar_fails(self):
        decision = admit(_record("jpeg", ("jpeg",), treatment="jpeg_q70"),
                         None, APPLICABILITY)
        assert decision["conditional_ok"] is False
        assert any("0.431" in r for r in decision["reasons"])
        assert decision["category"] == "reject"

    def test_the_weak_expert_cannot_carry_a_positive_sample(self):
        """frequency alone is corroboration; it may never be the whole story."""
        decision = admit(_record("frequency", ("freq",)), None, APPLICABILITY)
        assert decision["category"] != "positive"
        assert any("passed the bar on its own" in r for r in decision["reasons"])

    def test_a_weak_expert_alongside_a_strong_one_is_allowed(self):
        tool = _record("noise+frequency", ("noise", "freq"),
                       brier=0.01, verdict="Fake")
        baseline = _record("no-tool", brier=0.4)
        decision = admit(tool, baseline, APPLICABILITY)
        assert decision["category"] == "positive"

    def test_retired_experts_are_refused(self):
        decision = admit(_record("ela", ("ela",)), None, APPLICABILITY)
        assert decision["category"] == "reject"
        assert any("retired expert" in r for r in decision["reasons"])


class TestGainLayer:
    def test_a_brier_improvement_admits_a_correct_trajectory(self):
        tool = _record("noise", ("noise",), brier=0.05, verdict="Fake")
        baseline = _record("no-tool", brier=0.30)
        decision = admit(tool, baseline, APPLICABILITY)
        assert decision["category"] == "positive"
        assert decision["risk_reduced"] is True
        assert decision["brier_delta"] == pytest.approx(-0.25)

    def test_a_hair_of_improvement_is_not_a_gain(self):
        tool = _record("noise", ("noise",), brier=0.30 - RISK_EPSILON / 2)
        baseline = _record("no-tool", brier=0.30)
        decision = admit(tool, baseline, APPLICABILITY)
        assert decision["risk_reduced"] is False

    def test_an_nll_improvement_also_counts(self):
        tool = _record("noise", ("noise",), brier=0.30, nll=0.05)
        baseline = _record("no-tool", brier=0.30, nll=0.50)
        assert admit(tool, baseline, APPLICABILITY)["risk_reduced"] is True

    def test_a_confident_error_is_rejected_however_the_scores_look(self):
        tool = _record("noise", ("noise",), verdict="Real", ground_truth="Fake",
                       confidence=0.95, brier=0.01)
        baseline = _record("no-tool", brier=0.4)
        decision = admit(tool, baseline, APPLICABILITY)
        assert decision["category"] == "reject"
        assert any("confident error" in r for r in decision["reasons"])

    def test_correctness_is_required_even_when_risk_falls(self):
        tool = _record("noise", ("noise",), verdict="Real", ground_truth="Fake",
                       confidence=0.5, brier=0.01)
        baseline = _record("no-tool", brier=0.4)
        decision = admit(tool, baseline, APPLICABILITY)
        assert decision["category"] == "reject"
        assert any("still wrong" in r for r in decision["reasons"])

    def test_a_missing_baseline_cannot_support_a_positive_sample(self):
        decision = admit(_record("noise", ("noise",), brier=0.01), None, APPLICABILITY)
        assert decision["category"] != "positive"
        assert any("no no-tool baseline" in r for r in decision["reasons"])

    def test_cost_is_recorded_for_the_admitted_and_the_rejected(self):
        tool = _record("noise", ("noise",), brier=0.05, cost=4.0)
        baseline = _record("no-tool", brier=0.30, cost=1.0)
        decision = admit(tool, baseline, APPLICABILITY)
        assert decision["cost_delta"] == pytest.approx(3.0)


class TestHonestAbstention:
    def test_abstaining_where_the_baseline_was_confidently_wrong_is_kept(self):
        tool = _record("noise", ("noise",), verdict="Uncertain", confidence=0.5,
                       brier=0.25)
        baseline = _record("no-tool", verdict="Real", confidence=0.95, brier=0.4)
        decision = admit(tool, baseline, APPLICABILITY)
        assert decision["category"] == "honest_abstention"

    def test_abstaining_where_the_baseline_also_abstained_is_kept(self):
        tool = _record("noise", ("noise",), verdict="Uncertain", brier=0.25)
        baseline = _record("no-tool", verdict="Uncertain", brier=0.25)
        assert admit(tool, baseline, APPLICABILITY)["category"] == "honest_abstention"

    def test_abstaining_where_the_baseline_was_right_is_not_kept(self):
        tool = _record("noise", ("noise",), verdict="Uncertain", brier=0.25)
        baseline = _record("no-tool", verdict="Fake", confidence=0.9, brier=0.01)
        assert admit(tool, baseline, APPLICABILITY)["category"] == "reject"


class TestRun:
    def _corpus(self):
        return [
            _record("no-tool", brier=0.30),
            _record("noise", ("noise",), brier=0.05),
            _record("jpeg", ("jpeg",), treatment="jpeg_q70", brier=0.05),
        ]

    def test_it_decides_every_trajectory(self):
        report = run(self._corpus(), APPLICABILITY)
        assert report["trajectories"] == 3
        assert report["variants"] == 1
        assert report["categories"]["positive"] == 1
        assert report["categories"]["reject"] == 2

    def test_it_reports_cost_for_admitted_versus_rejected(self):
        report = run(self._corpus(), APPLICABILITY)
        assert report["admitted_mean_cost_delta"] is not None
        assert report["rejected_mean_cost_delta"] is not None

    def test_per_policy_summary_counts_positives(self):
        report = run(self._corpus(), APPLICABILITY)
        assert report["per_policy"]["noise"]["positive"] == 1
        assert report["per_policy"]["jpeg"]["positive"] == 0


class TestLoading:
    def test_it_reads_json_records_and_skips_junk(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps(_record("noise", ("noise",))),
                                         encoding="utf-8")
        (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
        (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
        (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

        records = load_trajectories(str(tmp_path))
        assert len(records) == 1
        assert records[0]["policy"] == "noise"

    def test_pairing_groups_policies_by_variant(self):
        records = [_record("no-tool", variant="v1"), _record("noise", ("noise",), variant="v1"),
                   _record("no-tool", variant="v2")]
        paired = pair_by_variant(records)
        assert set(paired) == {"v1", "v2"}
        assert set(paired["v1"]) == {"no-tool", "noise"}
