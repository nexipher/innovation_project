"""Tests for the offline halting replay (G3-c)."""

import json

import pytest

from scripts.replay_halting_g3 import (
    brier,
    ece,
    expert_weights,
    load_traces,
    model_candidate,
    summarise,
)


def _row(gt, recorded, v2, posterior=0.8, conflict=0.0, candidate="Fake",
         overrides=False):
    return {
        "ground_truth": gt,
        "recorded_verdict": recorded,
        "v2_verdict": v2,
        "v2_posterior": posterior,
        "conflict_score": conflict,
        "model_candidate": candidate,
        "v2_overrides_model": overrides,
        "recorded_calls": 2,
        "recorded_turns": 3,
        "recorded_reason": "verdict_output",
        "v2_reason": "no_expected_gain",
    }


class TestModelCandidate:
    def test_reads_the_last_verdict(self):
        trace = {"conversations": [
            {"from": "gpt", "value": '<verdict>{"verdict": "Real"}</verdict>'},
            {"from": "user", "value": "..."},
            {"from": "gpt", "value": '<verdict>{"verdict": "Fake"}</verdict>'},
        ]}
        assert model_candidate(trace) == "Fake"

    def test_none_when_the_model_never_concluded(self):
        assert model_candidate({"conversations": [{"from": "gpt", "value": "no tag"}]}) is None


class TestSummarise:
    def test_accuracy_ignores_abstentions(self):
        rows = [
            _row("Fake", "Fake", "Fake"),      # v1 right, v2 right
            _row("Real", "Fake", "Uncertain"),  # v1 wrong, v2 abstains
        ]
        summary = summarise(rows)
        assert summary["v1"]["accuracy"] == pytest.approx(0.5)
        assert summary["v2"]["accuracy"] == pytest.approx(1.0)
        assert summary["v2"]["abstained"] == 1

    def test_override_and_acceptance_counts(self):
        rows = [
            _row("Fake", "Fake", "Fake", candidate="Fake"),
            _row("Real", "Real", "Uncertain", candidate="Real", overrides=True),
        ]
        summary = summarise(rows)
        assert summary["override"]["model_candidates"] == 2
        assert summary["override"]["v2_overrides"] == 1
        assert summary["override"]["candidate_would_be_accepted"] == 1

    def test_conflict_sessions_are_counted_on_both_sides(self):
        rows = [
            _row("Fake", "Fake", "Uncertain", conflict=1.0, overrides=True),
            _row("Real", "Real", "Real", conflict=0.0),
        ]
        summary = summarise(rows)
        assert summary["conflict"]["sessions_with_conflict"] == 1
        assert summary["conflict"]["v1_labelled_those"] == 1
        assert summary["conflict"]["v2_abstained_those"] == 1

    def test_posterior_is_scored_as_the_probability_of_fake(self):
        # `posterior` is P(Fake), so a confidently-Real posterior on a Real
        # sample and a confidently-Fake one on a Fake sample both score 0.01.
        rows = [_row("Real", "Real", "Real", posterior=0.1),
                _row("Fake", "Fake", "Fake", posterior=0.9)]
        summary = summarise(rows)
        assert summary["v2"]["brier"] == pytest.approx(0.01)

    def test_abstentions_are_still_scored(self):
        """An abstaining posterior must be honest, not excused."""
        rows = [_row("Fake", "Uncertain", "Uncertain", posterior=0.5)]
        summary = summarise(rows)
        assert summary["v2"]["brier"] == pytest.approx(0.25)


class TestCalibrationMetrics:
    def test_ece_needs_ten_records(self):
        assert ece([0.9] * 5, [1] * 5) is None
        assert brier([], []) is None

    def test_perfect_calibration_scores_zero_brier(self):
        assert brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0)

    def test_ece_rewards_honest_confidence(self):
        honest = ece([1.0] * 5 + [0.0] * 5, [1] * 5 + [0] * 5)
        assert honest == pytest.approx(0.0, abs=1e-9)


class TestLoadTraces:
    def _write(self, directory, name, payload):
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_keeps_only_qwen_sessions_with_evidence_and_truth(self, tmp_path):
        base = {
            "ground_truth": "Fake",
            "evidence_chain": [{"source": "jpeg_expert"}],
            "metadata": {"mock_mode": "qwen_real"},
            "conversations": [],
            "final_verdict": {"verdict": "Fake"},
        }
        self._write(tmp_path, "forensic_sft_session_20260924_120000_a.json", base)
        self._write(tmp_path, "forensic_sft_session_20260924_120001_b.json",
                    {**base, "metadata": {"mock_mode": "two_calls"}})
        self._write(tmp_path, "forensic_sft_session_20260924_120002_c.json",
                    {**base, "evidence_chain": []})
        self._write(tmp_path, "forensic_sft_session_20260924_120003_d.json",
                    {**base, "ground_truth": None})
        self._write(tmp_path, "unrelated.json", base)

        traces = load_traces(sessions_dir=str(tmp_path))
        assert len(traces) == 1
        assert traces[0]["_path"].endswith("_a.json")


class TestExpertWeights:
    def test_missing_table_yields_zero_weights(self):
        assert expert_weights(None, ["noise_expert"]) == {"noise_expert": 0.0}

    def test_weights_come_from_the_calibration_table(self):
        class Table:
            @staticmethod
            def expert_entry(name):
                return {"separation_polarity_corrected": 0.972}

        assert expert_weights(Table(), ["jpeg_expert"])["jpeg_expert"] == \
            pytest.approx(0.944)
