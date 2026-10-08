"""Tests for review recording and the reviewed training set (G4-f → G4-g)."""

import json

import pytest

from scripts.apply_review import (
    OUTPUT_FILES,
    apply,
    assert_not_frozen,
    class_weights,
    load_samples,
)
from scripts.record_review import DECISIONS, load_dispositions, record, summary


def _sample(sample_id, truth="Fake", bucket="positive"):
    return {
        "id": sample_id, "ground_truth": truth, "type": bucket,
        "conversations": [], "evidence_chain": [],
        "final_verdict": {"verdict": "Fake", "confidence": 0.9},
        "metadata": {"trajectory_id": "noise__v", "tools_served": ["noise"]},
    }


@pytest.fixture
def final_v2(tmp_path):
    directory = tmp_path / "final_v2"
    directory.mkdir()
    (directory / "sft_tool_positive.json").write_text(
        json.dumps([_sample("s1"), _sample("s2")]), encoding="utf-8")
    (directory / "sft_no_tool_positive.json").write_text(
        json.dumps([_sample("s3", truth="Real", bucket="no_tool_positive")]),
        encoding="utf-8")
    (directory / "sft_honest_abstention.json").write_text(
        json.dumps([_sample("s4", truth="Real", bucket="honest_abstention")]),
        encoding="utf-8")
    return str(directory)


class TestRecording:
    def test_it_appends_a_decision(self, tmp_path, final_v2):
        path = str(tmp_path / "d.jsonl")
        entry = record("s1", "hj", "accept", "看起来对", path, final_v2)
        assert entry["decision"] == "accept"
        assert load_dispositions(path)["s1"]["reviewer"] == "hj"

    def test_the_latest_decision_wins(self, tmp_path, final_v2):
        path = str(tmp_path / "d.jsonl")
        record("s1", "hj", "accept", "", path, final_v2)
        record("s1", "hj", "reject", "再看了一遍", path, final_v2)
        assert load_dispositions(path)["s1"]["decision"] == "reject"

    def test_every_decision_keeps_its_history(self, tmp_path, final_v2):
        path = str(tmp_path / "d.jsonl")
        record("s1", "hj", "accept", "", path, final_v2)
        record("s1", "hj", "revise", "", path, final_v2)
        lines = open(path, encoding="utf-8").read().strip().split("\n")
        assert len(lines) == 2

    def test_an_unknown_decision_is_refused(self, tmp_path, final_v2):
        with pytest.raises(ValueError, match="unknown decision"):
            record("s1", "hj", "maybe", "", str(tmp_path / "d.jsonl"), final_v2)

    def test_a_missing_reviewer_is_refused(self, tmp_path, final_v2):
        with pytest.raises(ValueError, match="reviewer"):
            record("s1", "", "accept", "", str(tmp_path / "d.jsonl"), final_v2)

    def test_a_typo_in_the_sample_id_is_refused(self, tmp_path, final_v2):
        """A decision on a typo'd id would leave the real sample undecided."""
        with pytest.raises(ValueError, match="unknown sample id"):
            record("s9", "hj", "accept", "", str(tmp_path / "d.jsonl"), final_v2)

    def test_the_vocabulary_matches_the_g0_audit(self):
        assert set(DECISIONS) == {"accept", "format_only", "revise", "reject"}


class TestSummary:
    def test_it_counts_progress_by_bucket(self, tmp_path, final_v2):
        path = str(tmp_path / "d.jsonl")
        record("s1", "hj", "accept", "", path, final_v2)
        report = summary(load_dispositions(path), final_v2)
        assert report["samples"] == 4
        assert report["decided"] == 1
        assert report["pending"] == 3
        assert report["complete"] is False
        assert report["by_bucket_and_decision"]["positive|accept"] == 1
        assert report["by_bucket_and_decision"]["positive|pending"] == 1

    def test_it_reports_completion(self, tmp_path, final_v2):
        path = str(tmp_path / "d.jsonl")
        for sample_id in ("s1", "s2", "s3", "s4"):
            record(sample_id, "hj", "accept", "", path, final_v2)
        report = summary(load_dispositions(path), final_v2)
        assert report["complete"] is True
        assert report["by_decision"] == {"accept": 4}


class TestApply:
    def _dispositions(self, tmp_path, final_v2, decisions):
        path = str(tmp_path / "d.jsonl")
        for sample_id, decision in decisions.items():
            record(sample_id, "hj", decision, "", path, final_v2)
        return load_dispositions(path)

    def test_it_refuses_an_incomplete_review(self, tmp_path, final_v2):
        dispositions = self._dispositions(tmp_path, final_v2, {"s1": "accept"})
        with pytest.raises(SystemExit, match="undecided"):
            apply(dispositions, final_v2, str(tmp_path / "out"))

    def test_partial_marks_the_output_provisional(self, tmp_path, final_v2):
        dispositions = self._dispositions(tmp_path, final_v2, {"s1": "accept"})
        metadata = apply(dispositions, final_v2, str(tmp_path / "out"),
                         allow_partial=True)
        assert metadata["provisional"] is True
        assert metadata["undecided"] == 3
        assert metadata["counts"]["accept"] == 1

    def test_it_sorts_samples_into_the_four_sets(self, tmp_path, final_v2):
        dispositions = self._dispositions(tmp_path, final_v2, {
            "s1": "accept", "s2": "reject", "s3": "format_only", "s4": "revise"})
        out = tmp_path / "out"
        metadata = apply(dispositions, final_v2, str(out))
        assert metadata["counts"] == {"accept": 1, "format_only": 1,
                                      "revise": 1, "reject": 1}
        for decision, filename in OUTPUT_FILES.items():
            rows = json.loads((out / filename).read_text(encoding="utf-8"))
            assert len(rows) == 1
            assert rows[0]["review"]["decision"] == decision

    def test_the_accepted_set_carries_its_review(self, tmp_path, final_v2):
        dispositions = self._dispositions(tmp_path, final_v2, {"s1": "accept"})
        apply(dispositions, final_v2, str(tmp_path / "out"), allow_partial=True)
        rows = json.loads((tmp_path / "out" / "sft_accepted.json").read_text(encoding="utf-8"))
        assert rows[0]["review"]["reviewer"] == "hj"

    def test_weights_balance_the_accepted_set(self, tmp_path, final_v2):
        dispositions = self._dispositions(tmp_path, final_v2, {
            "s1": "accept", "s2": "accept", "s3": "accept"})
        metadata = apply(dispositions, final_v2, str(tmp_path / "out"),
                         allow_partial=True)
        # Two Fake, one Real accepted → Real needs twice the weight.
        assert metadata["accepted_labels"] == {"Fake": 2, "Real": 1}
        assert metadata["accepted_training_weights"] == {"Fake": 1.0, "Real": 2.0}

    def test_class_weights_of_an_empty_set(self):
        import collections
        assert class_weights(collections.Counter()) == {}

    def test_writing_into_a_frozen_set_is_refused(self):
        from scripts.apply_review import FROZEN_DIRS

        for frozen in FROZEN_DIRS:
            with pytest.raises(ValueError, match="frozen"):
                assert_not_frozen(frozen)
            with pytest.raises(ValueError):
                assert_not_frozen(frozen + "/nested")

    def test_load_samples_reads_all_buckets(self, final_v2):
        assert len(load_samples(final_v2)) == 4
