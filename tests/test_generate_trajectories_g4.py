"""Tests for the G4-b trajectory generator (CPU paths only)."""

import json
import math

import pytest

from scripts.generate_trajectories_g4 import (
    APPLICABILITY_THRESHOLD,
    EXPERT_BY_TOOL,
    TOOL_POLICIES,
    aggregate,
    build_experts,
    load_applicability,
    plan_trajectories,
    score_record,
    tool_applicable,
)

REAL_IMAGE = "dataset/Real/002baac0-bacd-496c-981c-a4a9d66b8472.jpg"


def _report():
    """A miniature G2-b report: jpeg inverted, noise aligned, freq weak."""
    return {
        "experts": {
            "jpeg": {
                "semantics_aligned": False,
                "cell_metrics": {
                    "png": {"auroc_fake_vs_real": 0.028},
                    "jpeg_q70": {"auroc_fake_vs_real": 0.569},
                },
            },
            "noise": {
                "semantics_aligned": False,
                "cell_metrics": {"png": {"auroc_fake_vs_real": 0.155}},
            },
            "frequency_v2": {
                "semantics_aligned": True,
                "cell_metrics": {"png": {"auroc_fake_vs_real": 0.556}},
            },
        }
    }


@pytest.fixture
def applicability(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps(_report()), encoding="utf-8")
    return load_applicability(str(path))


class TestApplicability:
    def test_inverted_experts_are_corrected(self, applicability):
        # A high jpeg metric means Real, so the separation it offers is 1-auroc.
        assert applicability["jpeg"]["png"] == pytest.approx(0.972)
        assert applicability["noise"]["png"] == pytest.approx(0.845)

    def test_aligned_experts_are_taken_as_measured(self, applicability):
        assert applicability["frequency_v2"]["png"] == pytest.approx(0.556)

    def test_applicable_only_above_the_threshold(self, applicability):
        assert tool_applicable("jpeg", "png", applicability)[0] is True
        ok, reason = tool_applicable("jpeg", "jpeg_q70", applicability)
        assert ok is False
        assert "0.431" in reason and str(APPLICABILITY_THRESHOLD) in reason

    def test_unmeasured_cells_are_refused_not_assumed(self, applicability):
        ok, reason = tool_applicable("jpeg", "native", applicability)
        assert ok is False
        assert "no measurement for cell" in reason

    def test_unknown_expert_is_refused(self, applicability):
        ok, reason = tool_applicable("ela", "png", applicability)
        assert ok is False

    def test_the_weak_expert_is_not_applicable_anywhere(self, applicability):
        assert tool_applicable("freq", "png", applicability)[0] is False


class TestPolicies:
    def test_no_tool_policy_serves_nothing_and_forbids_exploration(self):
        policy = TOOL_POLICIES["no-tool"]
        assert policy["tools"] == ()
        assert policy["allow_exploration"] is False
        assert policy["client"] == "baseline"

    def test_every_tool_maps_to_a_registered_expert(self):
        for policy in TOOL_POLICIES.values():
            for tool in policy["tools"]:
                assert tool in EXPERT_BY_TOOL

    def test_only_the_tool_policies_require_applicability(self):
        requiring = {name for name, p in TOOL_POLICIES.items() if p.get("requires")}
        assert requiring == {"jpeg", "noise+jpeg"}

    def test_build_experts_serves_only_the_requested_tools(self):
        experts = build_experts(("noise",))
        assert list(experts) == ["noise_expert"]

    def test_a_no_tool_session_gets_no_experts(self):
        assert build_experts(()) == {}


class TestPlanning:
    def _variants(self, treatment):
        return {f"v_{treatment}": {
            "variant_id": f"v_{treatment}", "source_id": "ADM/x", "label": "Fake",
            "generator": "ADM", "split": "train", "treatment": treatment,
            "container": "jpg", "path": f"x_{treatment}.jpg", "resolution": [32, 32],
        }}

    def test_jpeg_policies_are_gated_out_where_the_expert_does_not_separate(
            self, applicability):
        planned, skipped = plan_trajectories(
            self._variants("jpeg_q70"), ["no-tool", "jpeg", "noise+jpeg", "noise"],
            applicability)
        assert {p for _, p in planned} == {"no-tool", "noise"}
        assert {s["policy"] for s in skipped} == {"jpeg", "noise+jpeg"}
        assert all("0.431" in s["reason"] for s in skipped)

    def test_jpeg_policies_survive_where_it_does_separate(self, applicability):
        planned, skipped = plan_trajectories(
            self._variants("png"), ["jpeg", "noise+jpeg"], applicability)
        assert len(planned) == 2
        assert skipped == []

    def test_a_weak_but_unconditional_expert_is_still_generated(self, applicability):
        """`frequency` is generated on purpose — G4-c decides its admission."""
        planned, skipped = plan_trajectories(
            self._variants("png"), ["frequency"], applicability)
        assert len(planned) == 1
        assert skipped == []

    def test_planning_covers_every_variant_policy_pair(self, applicability):
        variants = {**self._variants("png"), "w": self._variants("png")["v_png"]}
        planned, _ = plan_trajectories(variants, ["no-tool", "noise"], applicability)
        assert len(planned) == len(variants) * 2


class TestScoring:
    def _record(self, **overrides):
        record = {
            "ground_truth": "Fake",
            "model_probability": 0.9,
            "posterior": 0.8,
        }
        record.update(overrides)
        return record

    def test_nll_and_brier_are_computed_for_both_probabilities(self):
        scores = score_record(self._record())
        assert scores["model_nll"] == pytest.approx(-math.log(0.9), abs=1e-4)
        assert scores["model_brier"] == pytest.approx(0.01)
        assert scores["posterior_nll"] == pytest.approx(-math.log(0.8), abs=1e-4)
        assert scores["posterior_brier"] == pytest.approx(0.04)

    def test_a_real_sample_scores_the_complement(self):
        scores = score_record(self._record(ground_truth="Real", model_probability=0.1))
        assert scores["model_brier"] == pytest.approx(0.01)

    def test_a_missing_model_probability_skips_only_the_model_scores(self):
        scores = score_record(self._record(model_probability=None))
        assert "model_nll" not in scores
        assert "posterior_brier" in scores

    def test_posterior_error_is_recorded(self):
        scores = score_record(self._record(posterior=0.3))
        assert scores["posterior_error"] == pytest.approx(0.7)


class TestAggregate:
    def _row(self, policy, verdict="Uncertain", calls=1, posterior_brier=0.25):
        return {
            "policy": policy, "final_verdict": verdict,
            "counters": {"expert_calls": calls, "model_turns": 2},
            "scores": {"posterior_brier": posterior_brier, "model_brier": 0.3},
        }

    def test_summarises_per_policy(self):
        summary = aggregate([self._row("noise"), self._row("noise", "Fake"),
                             self._row("no-tool", "Real", calls=0)])
        assert summary["noise"]["n"] == 2
        assert summary["noise"]["uncertain_rate"] == pytest.approx(0.5)
        assert summary["no-tool"]["mean_calls"] == 0
        assert summary["noise"]["mean_posterior_brier"] == pytest.approx(0.25)


class TestDryRunEndToEnd:
    """The whole path: manifest -> plan -> (mock) session -> record -> report."""

    def _manifest(self, tmp_path, treatment="native"):
        # The dataset image itself stands in for a variant: the dry run only
        # needs a readable file at the manifest's path.
        variant_path = REAL_IMAGE
        manifest = {"variants": {variant_path: {
            "variant_id": f"Real_002baac0_{treatment}",
            "source_id": "Real/002baac0", "label": "Real", "generator": "Real",
            "split": "train", "resolution": [24, 32], "treatment": treatment,
            "container": "jpg", "quality": None,
        }}}
        path = tmp_path / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return str(path)

    def test_it_writes_one_record_per_planned_pair(self, tmp_path, monkeypatch, capsys):
        import scripts.generate_trajectories_g4 as module

        out = tmp_path / "trajectories"
        monkeypatch.setattr(module, "REPORT_PATH", str(tmp_path / "report.json"))
        monkeypatch.setattr("sys.argv", [
            "generate_trajectories_g4.py", "--dry-run",
            "--manifest", self._manifest(tmp_path),
            "--policies", "no-tool", "noise",
            "--output-dir", str(out),
        ])
        module.main()

        written = sorted(p.name for p in out.glob("*.json"))
        assert written == ["Real_002baac0_native__no-tool.json",
                           "Real_002baac0_native__noise.json"]
        record = json.loads((out / written[0]).read_text(encoding="utf-8"))
        assert record["policy"] == "no-tool"
        assert record["tools_served"] == []
        assert record["ground_truth"] == "Real"
        assert "scores" in record

    def test_a_second_run_skips_what_is_already_there(self, tmp_path, monkeypatch, capsys):
        import scripts.generate_trajectories_g4 as module

        out = tmp_path / "trajectories"
        monkeypatch.setattr(module, "REPORT_PATH", str(tmp_path / "report.json"))
        argv = ["generate_trajectories_g4.py", "--dry-run",
                "--manifest", self._manifest(tmp_path),
                "--policies", "no-tool", "--output-dir", str(out)]
        monkeypatch.setattr("sys.argv", argv)
        module.main()
        module.main()

        report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
        assert report["already_present"] == 1
        assert report["written"] == 0
