"""
Tests for batch identity (G7 entry 4).

A batch is a directory plus a fingerprint, and the fingerprint is what makes
"these reviews are about this content" checkable rather than assumed.  The
properties worth pinning: the digest follows the training content, the batch
fingerprint follows all of it, and the legacy 237 keep the paths they were
reviewed under.
"""

import json
import os

import sys

import pytest

from config import PROJECT_ROOT
from utils.batch import (
    LEGACY_BATCH,
    batch_fingerprint,
    batch_paths,
    content_sha256,
    load_samples,
    write_manifest,
)


def _sample(sample_id, text="<verdict>Fake</verdict>"):
    return {"id": sample_id, "ground_truth": "Fake", "type": "positive",
            "conversations": [{"from": "gpt", "value": text}],
            "evidence_chain": [], "metadata": {"policy": "probe"}}


class TestContentDigest:
    def test_the_same_content_hashes_the_same(self):
        assert content_sha256(_sample("a")) == content_sha256(_sample("a"))

    def test_changed_dialogue_changes_the_digest(self):
        """The rendered conversation is what gets trained — it is the content."""
        first = content_sha256(_sample("a"))
        second = content_sha256(_sample("a", "<verdict>Real</verdict>"))
        assert first != second

    def test_the_review_block_does_not_participate(self):
        sample = _sample("a")
        annotated = {**sample, "review": {"reviewer": "hj", "decision": "accept"}}
        assert content_sha256(sample) == content_sha256(annotated)

    def test_the_bucket_file_does_not_participate(self):
        """Moving a sample between bucket files is bookkeeping, not content."""
        sample = _sample("a")
        moved = {**sample, "_bucket_file": "sft_honest_abstention.json"}
        assert content_sha256(sample) == content_sha256(moved)

    def test_a_changed_label_changes_the_digest(self):
        sample = _sample("a")
        assert content_sha256(sample) != content_sha256({**sample, "ground_truth": "Real"})


class TestBatchFingerprint:
    def test_order_does_not_matter(self):
        samples = [_sample("a"), _sample("b")]
        assert batch_fingerprint(samples) == batch_fingerprint(list(reversed(samples)))

    def test_one_changed_sample_changes_the_batch(self):
        samples = [_sample("a"), _sample("b")]
        changed = [_sample("a"), _sample("b", "<verdict>Real</verdict>")]
        assert batch_fingerprint(samples) != batch_fingerprint(changed)

    def test_it_is_short_and_stable(self):
        fingerprint = batch_fingerprint([_sample("a")])
        assert len(fingerprint) == 16
        assert fingerprint == batch_fingerprint([_sample("a")])


class TestBatchPaths:
    def test_the_legacy_batch_keeps_its_historical_paths(self):
        paths = batch_paths(LEGACY_BATCH)
        assert paths["samples"].endswith("sft_data/train/final_v2")
        assert paths["review"].endswith("sft_data/review")
        assert paths["reviewed"].endswith("sft_data/train/final_v2_reviewed")

    def test_a_new_batch_gets_its_own_directories(self):
        paths = batch_paths("g7_small")
        assert paths["samples"].endswith("sft_data/train/batches/g7_small")
        assert paths["review"].endswith("sft_data/review/g7_small")
        assert paths["reviewed"].endswith("sft_data/train/batches/g7_small_reviewed")
        # nothing a new batch writes can land in the legacy set
        assert paths["samples"] != batch_paths(LEGACY_BATCH)["samples"]
        assert paths["review"] != batch_paths(LEGACY_BATCH)["review"]

    def test_the_legacy_batch_still_reads_its_237_samples(self):
        samples = load_samples(LEGACY_BATCH)
        assert len(samples) == 237
        assert all("_bucket_file" in sample for sample in samples)


class TestManifest:
    def test_the_manifest_records_the_version_it_wrote(self, tmp_path):
        samples = [_sample("a"), _sample("b")]
        manifest = write_manifest("g7_small", str(tmp_path), samples=samples,
                                  generation={"policy": "probe", "seed": 1})
        assert manifest["fingerprint"] == batch_fingerprint(samples)
        assert manifest["counts"] == {"unfiled": {"Fake": 2}}
        with open(tmp_path / "manifest.json", encoding="utf-8") as handle:
            on_disk = json.load(handle)
        assert on_disk["fingerprint"] == manifest["fingerprint"]


class TestTheWorkstationCarriesTheBatch:
    def _build(self, tmp_path):
        from scripts.build_review_workstation import build

        samples_dir = tmp_path / "batch"
        samples_dir.mkdir()
        (tmp_path / "none").mkdir()          # empty trajectories dir
        (tmp_path / "none.json").write_text("{}", encoding="utf-8")
        samples = [_sample("a"), _sample("b")]
        (samples_dir / "sft_tool_positive.json").write_text(
            json.dumps(samples), encoding="utf-8")
        out = tmp_path / "review"
        build(str(samples_dir), str(out), trajectories_dir=str(tmp_path / "none"),
              variants_manifest=str(tmp_path / "none.json"), batch_id="g7_small")
        return samples, out

    def test_the_page_declares_which_batch_it_shows(self, tmp_path):
        samples, out = self._build(tmp_path)
        fingerprint = batch_fingerprint(samples)
        page = (out / "index.html").read_text(encoding="utf-8")
        assert "window.__batch" in page and fingerprint in page
        sample_page = (out / "samples" / "a.html").read_text(encoding="utf-8")
        assert fingerprint in sample_page

    def test_the_export_would_carry_it_too(self, tmp_path):
        _, out = self._build(tmp_path)
        page = (out / "index.html").read_text(encoding="utf-8")
        assert "batch_fingerprint: BATCH.fingerprint" in page


class TestGenerationIsolation:
    """
    A batch's whole pipeline is distinct, not just its reviews.

    The first version of `batch_paths` returned the *public* trajectories and
    variants directories for new batches, so a "new" batch would have read the
    237's inputs and written into them.  The plan asks for the generation to be
    isolated too, so the check belongs in a test rather than in a paragraph.
    """

    def test_no_new_batch_path_overlaps_a_legacy_one(self):
        legacy = set(batch_paths(LEGACY_BATCH).values())
        for batch_id in ("g7_small", "g7_small_2"):
            assert not (set(batch_paths(batch_id).values()) & legacy)

    def test_two_new_batches_do_not_share_anything(self):
        first = set(batch_paths("alpha").values())
        second = set(batch_paths("beta").values())
        assert not (first & second)

    def test_every_generation_input_is_inside_the_batch(self):
        paths = batch_paths("g7_small")
        for key in ("variants", "variants_dir", "trajectories", "admission", "samples"):
            assert "g7_small" in paths[key], key

    def test_the_generation_scripts_take_the_batch_id(self):
        """The id has to reach the scripts, or the paths are theory."""
        import subprocess

        for script in ("build_variants_g4.py", "generate_trajectories_g4.py",
                       "admit_trajectories_g4.py", "build_final_v2.py"):
            output = subprocess.run(
                [sys.executable, os.path.join(PROJECT_ROOT, "scripts", script), "--help"],
                capture_output=True, text=True, timeout=120).stdout
            assert "--batch" in output, script


class TestNoSharedOutputs:
    """Every script that writes a report must write it inside its batch."""

    def test_the_generator_writes_its_report_into_the_batch(self):
        from scripts.generate_trajectories_g4 import report_path_for

        legacy = batch_paths(LEGACY_BATCH)["trajectories"]
        new = batch_paths("g7_b1")["trajectories"]
        assert report_path_for(new) == os.path.join(new, "report.json")
        assert report_path_for(new) != report_path_for(legacy)
        # and the historical public report is not where a batch writes
        assert report_path_for(new) != os.path.join(PROJECT_ROOT, "sft_data",
                                                    "trajectories_report.json")

    def test_the_batch_report_on_disk_belongs_to_that_batch(self):
        """The move is only done if a reader can tell which batch it is."""
        path = os.path.join(PROJECT_ROOT, "sft_data", "trajectories_g7_b1",
                            "report.json")
        if not os.path.exists(path):
            pytest.skip("本批次报告尚未生成")
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        assert report.get("batch") == "g7_b1"

    def test_the_legacy_report_still_describes_the_legacy_batch(self):
        path = os.path.join(PROJECT_ROOT, "sft_data", "trajectories_report.json")
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["planned"] == 440 and report["written"] == 440
