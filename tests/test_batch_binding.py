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

    def test_the_page_shows_the_raw_turns_against_the_training_dialogue(self, tmp_path):
        """§4.17 R6: the reviewer can see what the converter did to the session."""
        import json as jsonlib

        from scripts.build_review_workstation import build

        samples_dir = tmp_path / "batch"
        samples_dir.mkdir()
        sample = _sample("a")
        sample["fidelity"] = {"model_reasoning_verbatim": True,
                              "reasoning_source": "model+bookkeeping",
                              "dialogue_audit": {"supervised_turns": 1,
                                                 "defect_count": 0,
                                                 "candidate_count": 0}}
        sample["conversations"] = [{"from": "user", "value": "分析。"},
                                   {"from": "gpt", "value": "四段式答案"}]
        sample.setdefault("metadata", {})["trajectory_id"] = "noise__ADM_x_png"
        (samples_dir / "sft_tool_positive.json").write_text(
            jsonlib.dumps([sample]), encoding="utf-8")

        trajectories = tmp_path / "trajectories"
        trajectories.mkdir()
        trace = {"conversations": [
            {"from": "user", "value": "分析。"},
            {"from": "gpt", "value": "原始助手轮：我倾向于伪造。"}]}
        trace_path = tmp_path / "trace.json"
        trace_path.write_text(jsonlib.dumps(trace), encoding="utf-8")
        record = {"trajectory_id": "noise__ADM_x_png", "trace_path": str(trace_path)}
        (trajectories / "t.json").write_text(jsonlib.dumps(record), encoding="utf-8")

        out = tmp_path / "review"
        build(str(samples_dir), str(out), trajectories_dir=str(trajectories),
              variants_manifest=str(tmp_path / "none.json"), batch_id="g7_small")
        page = (out / "samples" / "a.html").read_text(encoding="utf-8")
        assert "原始助手轮 vs 训练对话" in page
        assert "原始助手轮：我倾向于伪造。" in page
        assert "被四段式答案替换" in page
        assert "模型原文保留" in page

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

    def test_the_pages_have_no_nested_script_tags(self, tmp_path):
        """
        A `<script>` inside a `<script>` is a syntax error, and the browser
        stops running the whole block — the export button then does nothing at
        all, which is exactly what happened to the first batch's page.
        """
        _, out = self._build(tmp_path)
        for name in ("index.html", "samples/a.html"):
            page = (out / name).read_text(encoding="utf-8")
            assert "<script><script" not in page, name
            assert "</script></script>" not in page, name
            # the injected *assignment* appears once; the identifier is also
            # read by the script (BATCH = window.__batch || {})
            assert page.count("window.__batch = ") == 1, name
            assert page.count("window.__review") <= 1, name

    def test_the_export_button_is_wired(self, tmp_path):
        _, out = self._build(tmp_path)
        page = (out / "index.html").read_text(encoding="utf-8")
        assert 'id="export"' in page
        assert "exportButton.addEventListener('click', exportAll)" in page


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
        assert report_path_for(new, "20261011_170000").startswith(new)
        assert report_path_for(new, "20261011_170000") != \
            report_path_for(legacy, "20261011_170000")
        assert report_path_for(new, "a") != report_path_for(new, "b"), \
            "每次运行的报告不得互相覆盖"
        # and the historical public report is not where a batch writes
        assert report_path_for(new, "x") != os.path.join(PROJECT_ROOT, "sft_data",
                                                         "trajectories_report.json")

    def test_every_report_in_the_batch_says_which_run_it_was(self):
        """A reader must be able to tell which batch and which run it is."""
        import glob

        paths = glob.glob(os.path.join(PROJECT_ROOT, "sft_data", "trajectories_g7_b1",
                                       "report_*.json"))
        if not paths:
            pytest.skip("本批次报告尚未生成")
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                report = json.load(handle)
            assert report.get("batch") == "g7_b1", path
            assert report.get("policies"), path

    def test_the_legacy_report_still_describes_the_legacy_batch(self):
        path = os.path.join(PROJECT_ROOT, "sft_data", "trajectories_report.json")
        with open(path, encoding="utf-8") as handle:
            report = json.load(handle)
        assert report["planned"] == 440 and report["written"] == 440


class TestTheBuiltBatch:
    """The first training batch, checked the way the plan says to check it."""

    def test_the_manifest_fingerprint_describes_what_is_on_disk(self):
        """Recomputed, not hardcoded: a sample edited by hand shows up here."""
        from utils.batch import batch_fingerprint, load_samples

        path = os.path.join(PROJECT_ROOT, "sft_data", "train", "batches",
                            "g7_b1", "manifest.json")
        if not os.path.exists(path):
            pytest.skip("批次尚未构建")
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        samples = load_samples("g7_b1")
        assert manifest["samples"] == len(samples) == 145
        assert manifest["fingerprint"] == batch_fingerprint(samples)

    def test_no_sample_comes_from_a_reserved_partition(self):
        """The batch's sources are train-partition sources, by id, not by hope."""
        from utils.batch import load_samples

        samples = load_samples("g7_b1")
        if not samples:
            pytest.skip("批次尚未构建")
        with open(os.path.join(PROJECT_ROOT, "sft_data", "split_v2.json"),
                  encoding="utf-8") as handle:
            split = json.load(handle)
        sources = {split["sources"][key]["path"]
                   for key, entry in split["sources"].items()
                   if entry["split"] == "train"}
        test_paths = {e["path"] for e in split["sources"].values()
                      if e["split"] == "test"}
        assert all(sample["metadata"]["split"] == "train" for sample in samples)
        assert not (sources & test_paths)          # the guard itself
        # every sample's source is one of the train sources the split declares
        chosen = json.load(open(os.path.join(
            PROJECT_ROOT, "sft_data", "train", "batches", "g7_b1",
            "selection.json"), encoding="utf-8"))["chosen"]
        assert len(chosen) == 14
        assert all(split["sources"][key]["split"] == "train" for key in chosen)

    def test_near_chance_evidence_is_not_promoted_anywhere(self):
        """The promotion guard this batch was built to satisfy."""
        from utils.batch import load_samples
        from utils.weak_evidence import flags

        samples = load_samples("g7_b1")
        if not samples:
            pytest.skip("批次尚未构建")
        # the promotion flags are the ones the band-checker produces (no
        # `kind`); the ledger-derived families are asserted separately below
        promoted = {s["id"]: [f for f in flags(s) if "kind" not in f]
                    for s in samples}
        promoted = {key: value for key, value in promoted.items() if value}
        assert promoted == {}, promoted

    def test_the_extended_screener_finds_the_known_v2_defect(self):
        """
        This batch predates the repair: under v2 the policy owned the label,
        so samples whose own prose argues the other way are expected — the
        A-segment counted them (68 overridden).  The screener now finds that
        class at the sample layer; the batch is not re-scored by this test.
        """
        from utils.batch import load_samples
        from utils.weak_evidence import flags

        samples = load_samples("g7_b1")
        if not samples:
            pytest.skip("批次尚未构建")
        conflicts = [s["id"] for s in samples
                     if any(f.get("kind") == "label_report_conflict" for f in flags(s))]
        assert conflicts, "v2 批次应当体现被覆盖的模型结论"
        assert len(conflicts) < len(samples)

    def test_the_no_tool_arm_exists_in_both_classes(self):
        """The shortcut this batch must not teach: tools mean Fake."""
        from utils.batch import load_samples
        import collections

        samples = load_samples("g7_b1")
        if not samples:
            pytest.skip("批次尚未构建")
        by_bucket = collections.defaultdict(collections.Counter)
        for sample in samples:
            by_bucket[sample["_bucket_file"]][sample["ground_truth"]] += 1
        assert by_bucket["sft_tool_positive.json"]["Real"] >= 30
        assert by_bucket["sft_no_tool_positive.json"]["Real"] >= 20


class TestTheSamplesCarryTheirImages:
    """
    G7 entry 4: a sample knows which file the model was shown.

    The first conversion read the *public* variants manifest, so every sample's
    image_path came out empty and the review workstation showed "（未提供）"
    where the image belongs — the reviewer could not see what the model saw.
    """

    def test_every_sample_points_at_a_picture_that_exists(self):
        from utils.batch import load_samples

        samples = load_samples("g7_b1")
        if not samples:
            pytest.skip("批次尚未构建")
        empty = [s["id"] for s in samples if not s.get("image_path")]
        assert empty == [], empty
        missing = [s["image_path"] for s in samples
                   if not os.path.exists(os.path.join(PROJECT_ROOT, s["image_path"]))]
        assert missing == [], missing

    def test_the_image_is_the_variant_the_model_was_shown(self):
        """A q70 sample shows the q70 file, not the source."""
        from utils.batch import load_samples

        samples = load_samples("g7_b1")
        if not samples:
            pytest.skip("批次尚未构建")
        for sample in samples[:20]:
            assert "variants_g7_b1" in sample["image_path"], sample["id"]
            assert sample["metadata"]["treatment"] in sample["image_path"], sample["id"]

    def test_the_review_pages_render_the_images(self):
        page_dir = os.path.join(PROJECT_ROOT, "sft_data", "review", "g7_b1",
                                "samples")
        if not os.path.isdir(page_dir):
            pytest.skip("工作台尚未构建")
        pages = sorted(os.listdir(page_dir))[:10]
        for name in pages:
            with open(os.path.join(page_dir, name), encoding="utf-8") as handle:
                html = handle.read()
            assert "（未提供）" not in html, name
            assert "<img src=" in html, name
