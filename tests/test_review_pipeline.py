"""Tests for review recording and the reviewed training set (G4-f → G4-g)."""

import json
import os

import pytest

from scripts.apply_review import (
    OUTPUT_FILES,
    apply,
    assert_not_frozen,
    class_weights,
    load_samples,
)
from scripts.record_review import (

    DECISIONS,
    import_decisions,
    load_dispositions,
    parse_export,
    record,
    summary,
)

pytestmark = pytest.mark.slow

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

class TestImport:
    """The workstation saves in the browser; this is how it reaches the repo."""

    def _export(self, tmp_path, decisions, reviewer="hj", name="export.json",
                fingerprint=None, final_v2=None):
        """A workstation export — which, since G7, carries the batch version."""
        from scripts.record_review import batch_fingerprint_of

        if fingerprint is None and final_v2 is not None:
            fingerprint = batch_fingerprint_of(final_v2)
        path = tmp_path / name
        payload = {"reviewer": reviewer, "exported_at": "2026-10-09T10:00:00",
                   "decisions": decisions}
        if fingerprint is not None:
            payload["batch_fingerprint"] = fingerprint
        path.write_text(json.dumps(payload, ensure_ascii=False),
                        encoding="utf-8")
        return str(path)

    def test_it_appends_every_decided_row(self, tmp_path, final_v2):
        source = self._export(tmp_path, {
            "s1": {"decision": "accept", "notes": "证据一致", "saved_at": "2026-10-09T09:00:00"},
            "s2": {"decision": "reject", "notes": "容器当理由"},
        }, final_v2=final_v2)
        path = str(tmp_path / "d.jsonl")
        report = import_decisions(source, None, path, final_v2)
        assert report["imported"] == 2
        assert report["by_decision"] == {"accept": 1, "reject": 1}
        latest = load_dispositions(path)
        assert latest["s1"]["notes"] == "证据一致"
        assert latest["s1"]["reviewer"] == "hj"       # taken from the export
        assert latest["s2"]["decision"] == "reject"

    def test_an_untouched_row_is_not_a_decision(self, tmp_path, final_v2):
        source = self._export(tmp_path, {"s1": {"decision": "", "notes": ""},
                                         "s2": {"decision": "accept"}}, final_v2=final_v2)
        report = import_decisions(source, "hj", str(tmp_path / "d.jsonl"), final_v2)
        assert report["imported"] == 1

    def test_a_typo_costs_one_row_not_the_import(self, tmp_path, final_v2):
        source = self._export(tmp_path, {"s9": {"decision": "accept"},
                                         "s1": {"decision": "accept"}}, final_v2=final_v2)
        report = import_decisions(source, "hj", str(tmp_path / "d.jsonl"), final_v2)
        assert report["imported"] == 1
        assert report["unknown"] == ["s9"]

    def test_an_unknown_decision_is_skipped_and_reported(self, tmp_path, final_v2):
        source = self._export(tmp_path, {"s1": {"decision": "maybe"}}, final_v2=final_v2)
        report = import_decisions(source, "hj", str(tmp_path / "d.jsonl"), final_v2)
        assert report["imported"] == 0
        assert report["invalid"] == [("s1", "maybe")]

    def test_re_importing_the_same_export_changes_nothing(self, tmp_path, final_v2):
        source = self._export(tmp_path, {"s1": {"decision": "accept", "notes": "n"}}, final_v2=final_v2)
        path = str(tmp_path / "d.jsonl")
        import_decisions(source, "hj", path, final_v2)
        report = import_decisions(source, "hj", path, final_v2)
        assert report["imported"] == 0 and report["unchanged"] == 1
        assert len(open(path, encoding="utf-8").read().strip().split("\n")) == 1

    def test_a_changed_decision_is_appended(self, tmp_path, final_v2):
        path = str(tmp_path / "d.jsonl")
        import_decisions(self._export(tmp_path, {"s1": {"decision": "accept"}}, name="a.json", final_v2=final_v2),
                         "hj", path, final_v2)
        import_decisions(self._export(tmp_path, {"s1": {"decision": "revise", "notes": "漏了反证"}},
                                      name="b.json", final_v2=final_v2), "hj", path, final_v2)
        assert load_dispositions(path)["s1"]["decision"] == "revise"
        assert len(open(path, encoding="utf-8").read().strip().split("\n")) == 2

    def test_the_reviewer_can_come_from_the_command_line_instead(self, tmp_path, final_v2):
        source = self._export(tmp_path, {"s1": {"decision": "accept"}}, reviewer="", final_v2=final_v2)
        path = str(tmp_path / "d.jsonl")
        import_decisions(source, "hj", path, final_v2)
        assert load_dispositions(path)["s1"]["reviewer"] == "hj"

    def test_no_reviewer_anywhere_is_refused(self, tmp_path, final_v2):
        source = self._export(tmp_path, {"s1": {"decision": "accept"}}, reviewer="", final_v2=final_v2)
        with pytest.raises(ValueError, match="reviewer"):
            import_decisions(source, None, str(tmp_path / "d.jsonl"), final_v2)

    def test_a_filled_worklist_csv_is_read(self, tmp_path, final_v2):
        path = tmp_path / "worklist.csv"
        path.write_text(
            "sample_id,bucket,ground_truth,decision,notes,reviewer\n"
            "s1,positive,Fake,accept,看起来对,hj\n"
            "s2,positive,Fake,,\n"
            "s3,no_tool_positive,Real,format_only,结论勉强,hj\n",
            encoding="utf-8")
        # A hand-filled worklist carries no batch fingerprint, so it goes
        # through the explicit unversioned path (the legacy way of working).
        with pytest.raises(SystemExit, match="批次指纹"):
            import_decisions(str(path), None, str(tmp_path / "d.jsonl"), final_v2)
        report = import_decisions(str(path), None, str(tmp_path / "d.jsonl"), final_v2,
                                  allow_unversioned=True)
        assert report["imported"] == 2
        assert report["by_decision"] == {"accept": 1, "format_only": 1}

    def test_a_file_without_decisions_is_refused(self, tmp_path):
        path = tmp_path / "not_an_export.json"
        path.write_text(json.dumps({"hello": "world"}), encoding="utf-8")
        with pytest.raises(ValueError, match="decisions"):
            parse_export(str(path))

    def test_a_bare_id_map_is_accepted(self, tmp_path):
        path = tmp_path / "bare.json"
        path.write_text(json.dumps({"s1": {"decision": "accept", "notes": "x"}}),
                        encoding="utf-8")
        _, entries, batch = parse_export(str(path))
        assert entries["s1"]["decision"] == "accept"
        assert batch["batch_fingerprint"] == ""   # a pre-G7 export

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

class TestStaleDecisions:
    """G7 entry 4: a decision is about content, not about an id."""

    def _dispositions(self, tmp_path, final_v2, decisions):
        path = str(tmp_path / "d.jsonl")
        for sample_id, decision in decisions.items():
            record(sample_id, "hj", decision, "", path, final_v2)
        return load_dispositions(path)

    def test_a_decision_records_what_it_was_about(self, tmp_path, final_v2):
        entry = record("s1", "hj", "accept", "", str(tmp_path / "d.jsonl"), final_v2)
        assert entry["content_sha256"]
        assert entry["batch_fingerprint"]

    def test_regenerated_content_invalidates_the_decision(self, tmp_path, final_v2):
        dispositions = self._dispositions(tmp_path, final_v2,
                                          {"s1": "accept", "s2": "accept",
                                           "s3": "accept", "s4": "accept"})
        # s1 is regenerated: same id, different content (a new conclusion).
        path = os.path.join(final_v2, "sft_tool_positive.json")
        with open(path, encoding="utf-8") as handle:
            samples = json.load(handle)
        samples[0]["final_verdict"] = {"verdict": "Real", "confidence": 0.6}
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(samples, handle, ensure_ascii=False)

        metadata = apply(dispositions, final_v2, os.path.join(tmp_path, "out"),
                         allow_partial=True)
        assert [entry["sample_id"] for entry in metadata["stale_decisions"]] == ["s1"]
        assert metadata["undecided"] == 1
        assert metadata["counts"]["accept"] == 3

    def test_a_stale_set_cannot_be_written_without_saying_so(self, tmp_path, final_v2):
        dispositions = self._dispositions(tmp_path, final_v2,
                                          {"s1": "accept", "s2": "accept"})
        path = os.path.join(final_v2, "sft_tool_positive.json")
        with open(path, encoding="utf-8") as handle:
            samples = json.load(handle)
        samples[0]["ground_truth"] = "Real"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(samples, handle, ensure_ascii=False)
        with pytest.raises(SystemExit, match="undecided"):
            apply(dispositions, final_v2, os.path.join(tmp_path, "out"))

    def test_a_decision_from_another_batch_is_not_applied(self, tmp_path, final_v2):
        path = str(tmp_path / "d.jsonl")
        record("s1", "hj", "accept", "", path, final_v2, batch="other_batch")
        dispositions = load_dispositions(path)
        metadata = apply(dispositions, final_v2, os.path.join(tmp_path, "out"),
                         allow_partial=True)
        assert metadata["stale_decisions"][0]["sample_id"] == "s1"
        assert "另一个批次" in metadata["stale_decisions"][0]["why"]

    def test_require_hashes_refuses_legacy_records(self, tmp_path, final_v2):
        path = str(tmp_path / "d.jsonl")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"sample_id": "s1", "reviewer": "hj",
                                     "decision": "accept", "notes": "",
                                     "recorded_at": "2026-10-08T00:00:00"}) + "\n")
        dispositions = load_dispositions(path)
        with pytest.raises(SystemExit, match="内容哈希"):
            apply(dispositions, final_v2, os.path.join(tmp_path, "out"),
                  allow_partial=True, require_hashes=True)
