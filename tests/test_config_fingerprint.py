"""Tests for the experiment configuration fingerprint (G3-d)."""

import json

import pytest

from scripts.qwen_gain_baseline import load_completed
from utils import config_fingerprint as cf


class _Expert:
    def __init__(self, polarity=1, source_name="x_expert"):
        self.metric_polarity = polarity
        self.source_name = source_name


def _experts(polarity=1):
    return {"x_expert": _Expert(polarity)}


class TestCompute:
    def test_covers_the_things_that_change_a_measurement(self):
        fingerprint = cf.compute(_experts())
        assert set(fingerprint) >= {
            "experts", "prompts", "reliability_table", "rectifier",
            "tokenizer", "halting_policy", "git_commit", "digest",
        }
        assert fingerprint["experts"]["x_expert"]["polarity"] == 1

    def test_digest_is_stable_for_the_same_configuration(self):
        assert cf.compute(_experts())["digest"] == cf.compute(_experts())["digest"]

    def test_polarity_change_changes_the_digest(self):
        """The same expert read backwards is a different instrument."""
        assert cf.compute(_experts(1))["digest"] != cf.compute(_experts(-1))["digest"]

    def test_prompt_change_changes_the_digest(self, monkeypatch):
        import mllm.message_builder as builder

        before = cf.compute(_experts())["digest"]
        monkeypatch.setattr(builder, "FORENSIC_SYSTEM_PROMPT",
                            builder.FORENSIC_SYSTEM_PROMPT + "\nnew rule")
        assert cf.compute(_experts())["digest"] != before

    def test_expert_class_change_changes_the_digest(self):
        class Other(_Expert):
            pass

        assert cf.compute({"x": Other()})["digest"] != cf.compute({"x": _Expert()})["digest"]

    def test_registry_order_does_not_matter(self):
        first = cf.compute({"a": _Expert(source_name="a"), "b": _Expert(source_name="b")})
        second = cf.compute({"b": _Expert(source_name="b"), "a": _Expert(source_name="a")})
        assert first["digest"] == second["digest"]


class TestDifferences:
    def test_no_record_is_named(self):
        assert cf.differences(None, cf.compute(_experts())) == ["no fingerprint on record"]

    def test_reports_the_changed_components(self):
        stored = cf.compute(_experts(1))
        live = cf.compute(_experts(-1))
        assert cf.differences(stored, live) == ["experts"]

    def test_multiple_changes_are_listed(self, monkeypatch):
        import mllm.message_builder as builder

        stored = cf.compute(_experts())
        monkeypatch.setattr(builder, "BASELINE_SYSTEM_PROMPT", "different")
        changed = cf.differences(stored, cf.compute(_experts()))
        assert "prompts" in changed and "digest" not in changed


class TestResumeGuard:
    """The guard that replaces the manual --fresh."""

    def _report(self, path, fingerprint):
        payload = {
            "mode": "gpu", "per_cell": 5,
            "config_fingerprint": fingerprint,
            "conditions": {"rgb": {"records": [{"sample_id": "a"}]}},
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return str(path)

    def test_matching_fingerprint_resumes(self, tmp_path):
        fingerprint = cf.compute(_experts())
        path = self._report(tmp_path / "r.json", fingerprint)
        assert load_completed(path, "gpu", 5, fingerprint)["rgb"]

    def test_changed_configuration_starts_cold(self, tmp_path, capsys):
        path = self._report(tmp_path / "r.json", cf.compute(_experts(1)))
        result = load_completed(path, "gpu", 5, cf.compute(_experts(-1)))

        assert result == {}
        assert "configuration changed" in capsys.readouterr().out

    def test_reports_without_a_fingerprint_start_cold(self, tmp_path):
        """Every report written before G3-d is a cold start by construction."""
        payload = {"mode": "gpu", "per_cell": 5,
                   "conditions": {"rgb": {"records": [{"sample_id": "a"}]}}}
        path = tmp_path / "old.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert load_completed(str(path), "gpu", 5, cf.compute(_experts())) == {}

    def test_mode_and_size_still_guard(self, tmp_path):
        fingerprint = cf.compute(_experts())
        path = self._report(tmp_path / "r.json", fingerprint)
        assert load_completed(path, "dry_run", 5, fingerprint) == {}
        assert load_completed(path, "gpu", 15, fingerprint) == {}
