"""Tests for the G2-d four-condition gain comparison harness (CPU only).

The harness itself is GPU-gated; everything asserted here is the CPU-testable
scaffolding: stratified selection, metric definitions and the runner contract.
"""

import pytest

from mllm.mock_client import MockMLLMClient
from scripts.qwen_gain_baseline import (
    COMPARISON_CELLS,
    CONDITIONS,
    _auroc,
    _ece,
    _pseudo_probability,
    compute_metrics,
    run_condition,
    select_samples,
)
from state_machine.controller import ForensicStateMachine

REAL_PATH = "dataset/Real/002baac0-bacd-496c-981c-a4a9d66b8472.jpg"


def _sample(sample_id, cell, label):
    return {"sample_id": sample_id, "cell": cell, "label": label,
            "path": f"calibration/set/images/{sample_id}.png"}


class TestSelectSamples:
    def _manifest(self):
        samples = []
        for cell in COMPARISON_CELLS:
            for index in range(4):
                samples.append(_sample(f"{cell}_real{index}", cell, "Real"))
                samples.append(_sample(f"{cell}_fake{index}", cell, "Fake"))
        # A non-comparison cell must never enter the comparison.
        samples.append(_sample("real_blur_0", "real_blur", "Real"))
        return {"samples": samples}

    def test_balanced_per_cell(self):
        selected = select_samples(self._manifest(), 2)
        assert len(selected) == len(COMPARISON_CELLS) * 2 * 2

        counts = {}
        for sample in selected:
            counts[(sample["cell"], sample["label"])] = \
                counts.get((sample["cell"], sample["label"]), 0) + 1
        for cell in COMPARISON_CELLS:
            assert counts[(cell, "Real")] == 2
            assert counts[(cell, "Fake")] == 2

    def test_perturbation_cells_excluded(self):
        selected = select_samples(self._manifest(), 2)
        assert all(s["cell"] in COMPARISON_CELLS for s in selected)

    def test_returns_all_when_fewer_available(self):
        manifest = {"samples": [_sample("a", COMPARISON_CELLS[0], "Real")]}
        assert len(select_samples(manifest, 15)) == 1


class TestPseudoProbability:
    def test_fake_verdict_uses_confidence(self):
        assert _pseudo_probability({"verdict": "Fake", "confidence": 0.9}) == \
            pytest.approx(0.9)

    def test_real_verdict_is_complemented(self):
        assert _pseudo_probability({"verdict": "Real", "confidence": 0.8}) == \
            pytest.approx(0.2)

    def test_uncertain_is_uninformative(self):
        assert _pseudo_probability({"verdict": "Uncertain", "confidence": 0.0}) == 0.5

    def test_missing_confidence_defaults_to_zero(self):
        assert _pseudo_probability({"verdict": "Fake"}) == pytest.approx(0.0)


def _record(gt, verdict, confidence, cell="real_png"):
    return {"gt": gt, "verdict": verdict, "confidence": confidence, "cell": cell,
            "model_turns": 3, "expert_calls": 2}


class TestAuroc:
    def test_perfect_separation(self):
        records = [_record("Fake", "Fake", 0.9), _record("Fake", "Fake", 0.8),
                   _record("Fake", "Fake", 0.7),
                   _record("Real", "Real", 0.9), _record("Real", "Real", 0.8),
                   _record("Real", "Real", 0.7)]
        assert _auroc(records) == pytest.approx(1.0)

    def test_inverted_separation(self):
        """An expert that points the wrong way scores below chance."""
        records = [_record("Fake", "Real", 0.9), _record("Fake", "Real", 0.9),
                   _record("Fake", "Real", 0.9),
                   _record("Real", "Fake", 0.9), _record("Real", "Fake", 0.9),
                   _record("Real", "Fake", 0.9)]
        assert _auroc(records) == pytest.approx(0.0)

    def test_all_uncertain_is_chance(self):
        records = [_record("Fake", "Uncertain", 0.5)] * 3 + \
                  [_record("Real", "Uncertain", 0.5)] * 3
        assert _auroc(records) == pytest.approx(0.5)

    def test_insufficient_samples_return_none(self):
        assert _auroc([_record("Fake", "Fake", 0.9)] * 2) is None


class TestEce:
    def test_needs_ten_records(self):
        assert _ece([_record("Fake", "Fake", 0.9)] * 5) is None

    def test_overconfidence_is_measured(self):
        """confidence 0.9 everywhere, always correct → ECE = 0.1."""
        records = [_record("Fake", "Fake", 0.9)] * 10
        assert _ece(records) == pytest.approx(0.1)

    def test_perfectly_calibrated_verdicts(self):
        """Half at 0.0 / half at 1.0, each matching its outcome → ECE = 0."""
        records = [_record("Fake", "Fake", 1.0)] * 5 + \
                  [_record("Real", "Real", 1.0)] * 5
        assert _ece(records) == pytest.approx(0.0)


class TestComputeMetrics:
    def _records(self):
        return [
            _record("Fake", "Fake", 0.9, "fake_png"),
            _record("Fake", "Real", 0.8, "fake_png"),
            _record("Real", "Fake", 0.7, "real_png"),
            _record("Real", "Real", 0.6, "real_png"),
            _record("Fake", "Uncertain", 0.5, "fake_png"),
            _record("Real", "Real", 0.9, "real_png"),
        ]

    def test_core_metrics(self):
        metrics = compute_metrics(self._records())
        assert metrics["n"] == 6
        assert metrics["accuracy"] == pytest.approx(3 / 6)
        # precision 1/2, recall 1/3 → F1 = 0.4
        assert metrics["f1_fake"] == pytest.approx(0.4)
        assert metrics["uncertain_rate"] == pytest.approx(1 / 6)

    def test_per_cell_accuracy(self):
        metrics = compute_metrics(self._records())
        assert metrics["per_cell_accuracy"]["fake_png"] == pytest.approx(1 / 3)
        assert metrics["per_cell_accuracy"]["real_png"] == pytest.approx(2 / 3)

    def test_average_counters(self):
        metrics = compute_metrics(self._records())
        assert metrics["avg_model_turns"] == pytest.approx(3.0)
        assert metrics["avg_expert_calls"] == pytest.approx(2.0)

    def test_empty_input(self):
        assert compute_metrics([]) == {"n": 0}


class TestConditionSpec:
    def test_four_conditions_with_expected_injection(self):
        assert set(CONDITIONS) == {"rgb", "text", "image", "both"}
        assert CONDITIONS["rgb"]["injection"] == "none"
        assert CONDITIONS["text"]["injection"] == "text"
        assert CONDITIONS["image"]["injection"] == "image"
        assert CONDITIONS["both"]["injection"] == "text+image"

    def test_rgb_is_the_only_baseline_prompt_arm(self):
        baseline = [c for c, spec in CONDITIONS.items() if spec["client"] == "baseline"]
        assert baseline == ["rgb"]

    def test_dry_run_report_never_overwrites_the_gpu_report(self):
        from scripts.qwen_gain_baseline import report_path

        assert report_path(dry_run=True) != report_path(dry_run=False)
        assert report_path(dry_run=False).endswith("g2_gain_report.json")

    def test_injection_modes_are_valid_controller_modes(self):
        for spec in CONDITIONS.values():
            ForensicStateMachine(MockMLLMClient(mode="fast_verdict"),
                                 {}, evidence_injection=spec["injection"])

    def test_shipped_toolkit_matches_controller_dispatch(self):
        """The comparison must measure the toolkit the controller can dispatch."""
        from scripts.qwen_gain_baseline import _build_experts

        experts = _build_experts()
        dispatchable = set(ForensicStateMachine.EXPERT_KEY_MAP.values())
        assert dispatchable <= set(experts), dispatchable - set(experts)


class TestRunCondition:
    def _factory(self, seen):
        def factory(variant):
            seen.append(variant)
            return MockMLLMClient(mode="two_calls", seed=42)
        return factory

    def test_record_contract(self):
        seen = []
        samples = [{**_sample("real_0000_png", "real_png", "Real"), "path": REAL_PATH}]
        records = run_condition("text", samples, self._factory(seen), progress=False)

        assert seen == ["forensic"]
        assert len(records) == 1
        record = records[0]
        assert record["sample_id"] == "real_0000_png"
        assert record["gt"] == "Real"
        assert record["cell"] == "real_png"
        assert record["verdict"] in ("Real", "Fake", "Uncertain")
        assert record["model_turns"] >= 1
        assert record["elapsed_s"] >= 0

    def test_rgb_uses_the_baseline_client(self):
        seen = []
        samples = [{**_sample("real_0000_png", "real_png", "Real"), "path": REAL_PATH}]
        run_condition("rgb", samples, self._factory(seen), progress=False)
        assert seen == ["baseline"]

    def test_client_built_once_per_condition(self):
        """Constructing a client per sample would reload 16.6 GB of weights."""
        seen = []
        samples = [
            {**_sample(f"real_0000_png", "real_png", "Real"), "path": REAL_PATH},
            {**_sample(f"real_0001_png", "real_png", "Real"), "path": REAL_PATH},
            {**_sample(f"fake_0000_png", "fake_png", "Fake"), "path": REAL_PATH},
        ]
        records = run_condition("text", samples, self._factory(seen), progress=False)

        assert seen == ["forensic"], "client must be built once and reused"
        assert len(records) == 3

    def test_reused_client_is_reset_between_samples(self):
        """One client, many samples: each sample must start a fresh session."""

        class SpyClient:
            def __init__(self):
                self._inner = MockMLLMClient(mode="two_calls", seed=42)
                self.resets = 0

            def generate(self, image_path, history):
                return self._inner.generate(image_path, history)

            def reset(self):
                self.resets += 1
                self._inner.reset()

        spy = SpyClient()
        samples = [
            {**_sample("real_0000_png", "real_png", "Real"), "path": REAL_PATH},
            {**_sample("fake_0000_png", "fake_png", "Fake"), "path": REAL_PATH},
        ]
        run_condition("text", samples, lambda _variant: spy, progress=False)

        assert spy.resets == len(samples)
