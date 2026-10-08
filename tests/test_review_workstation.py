"""Tests for the G4-f review workstation."""

import csv
import json
import os

import pytest
from PIL import Image

from scripts.build_review_workstation import (
    CHECKLIST,
    build,
    double_review_reasons,
    load_samples,
    render_sample_page,
    stratified_sample,
)


def _sample(sample_id="f2_noise__ADM_x_png", verdict="Fake", confidence=0.9,
            truth="Fake", tools=("noise",), treatment="png", generator="ADM",
            answer="<observation>\n只有容器是 JPEG。\n</observation>",
            tokens=None):
    return {
        "id": sample_id,
        "image_path": "sft_data/variants/ADM_x_png.png",
        "ground_truth": truth,
        "source_model": generator,
        "final_verdict": {"verdict": verdict, "confidence": confidence,
                          "posterior": 0.9 if verdict == "Fake" else 0.1,
                          "policy_reasons": ["candidate_matches_posterior"]},
        "conversations": [{"from": "user", "value": "<image>\n请分析。"},
                          {"from": "gpt", "value": answer}],
        "evidence_chain": tokens if tokens is not None else [{
            "evidence_id": "E-1", "source": "noise_expert", "measurement_scope": "global",
            "raw_metric": 1.5, "region": "patch_coordinates_[0, 0, 10, 10]",
            "calibrated_likelihood": {"Real": 0.15, "Fake": 0.85}, "support": "AI-generated",
            "applicability": "inverted:high-metric-means-real",
            "applicability_conditions": "高 strength 对应 Real。",
            "diagnostic_region_image": "traces/evidence/s1/region_E-1.png",
            "visual_artifacts": ["traces/evidence/s1/noise_residual_map_E-1.png"],
        }],
        "metadata": {"trajectory_id": "noise__ADM_x_png", "policy": "noise",
                     "treatment": treatment, "split": "train", "tools_served": list(tools),
                     "admission_category": "positive",
                     "counters": {"weighted_cost": 2.0}},
        "type": "positive",
    }


def _trajectory(variant_id="ADM_x_png", policy="noise", conflict=0.0, brier=0.02):
    return {
        "trajectory_id": f"{policy}__{variant_id}", "policy": policy,
        "variant_id": variant_id, "conflict_score": conflict,
        "policy_reasons": ["candidate_matches_posterior"],
        "scores": {"posterior_brier": brier},
        "counters": {"weighted_cost": 2.0},
        "final_verdict": "Fake", "posterior": 0.9,
    }


class TestDoubleReviewTriggers:
    def test_an_uncertain_verdict_needs_a_second_reader(self):
        assert "uncertain" in double_review_reasons(
            _sample(verdict="Uncertain"), _trajectory())

    def test_multiple_tools_need_one(self):
        assert "multi_tool" in double_review_reasons(
            _sample(tools=("noise", "jpeg")), _trajectory())

    def test_a_confident_label_needs_one(self):
        assert "high_confidence" in double_review_reasons(
            _sample(confidence=0.9), _trajectory())

    def test_an_open_conflict_needs_one(self):
        assert "conflict" in double_review_reasons(
            _sample(), _trajectory(conflict=0.8, policy="noise"))

    def test_a_low_confidence_single_tool_answer_does_not(self):
        assert double_review_reasons(
            _sample(verdict="Fake", confidence=0.6), _trajectory()) == []


class TestStratifiedSample:
    def test_it_covers_every_stratum(self):
        samples = [_sample(sample_id=f"s{index}", treatment=t, generator=g)
                   for index, (t, g) in enumerate(
                       [("png", "ADM"), ("png", "ADM"), ("png", "SD14"),
                        ("jpeg_q70", "ADM")])]
        picked = stratified_sample(samples, 1)
        assert set(picked) == {"s0", "s2", "s3"}

    def test_it_takes_the_requested_number_per_stratum(self):
        samples = [_sample(sample_id=f"s{index}") for index in range(5)]
        assert len(stratified_sample(samples, 2)) == 2


class TestSamplePage:
    def _render(self, tmp_path, sample=None, source="", missing=None):
        page = str(tmp_path / "samples" / "x.html")
        os.makedirs(os.path.dirname(page), exist_ok=True)
        return render_sample_page(sample or _sample(), _trajectory(), _trajectory(policy="no-tool"),
                                  source, str(tmp_path / "index.html"), page,
                                  missing if missing is not None else set(), [])

    def test_it_shows_the_evidence_table_and_the_training_answer(self, tmp_path):
        page = self._render(tmp_path)
        assert "<th>evidence_id</th>" in page
        assert "E-1" in page
        assert "0.15 / 0.85" in page                     # calibrated likelihood
        assert "&lt;observation&gt;" in page               # escaped answer
        assert "no-tool 基线" in page

    def test_it_lists_every_checklist_item(self, tmp_path):
        import html as _html

        page = self._render(tmp_path)
        for name, text in CHECKLIST:
            # The checklist names tags, and tags are escaped in the page.
            assert _html.escape(text) in page

    def test_a_missing_image_is_flagged_and_counted(self, tmp_path):
        missing = set()
        page = self._render(tmp_path, missing=missing)
        assert "文件缺失" in page
        assert len(missing) > 0

    def test_an_answer_cannot_inject_markup(self, tmp_path):
        sample = _sample(answer="<observation>\n<script>alert(1)</script>\n</observation>")
        page = self._render(tmp_path, sample=sample)
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page


class TestBuildEndToEnd:
    def _fixture(self, tmp_path):
        """A one-sample final_v2 plus the manifest, split and trajectories."""
        project = tmp_path
        (project / "sft_data" / "train" / "final_v2").mkdir(parents=True)
        sample = _sample()
        (project / "sft_data" / "train" / "final_v2" / "sft_tool_positive.json").write_text(
            json.dumps([sample]), encoding="utf-8")
        (project / "sft_data" / "train" / "final_v2" / "sft_no_tool_positive.json").write_text(
            "[]", encoding="utf-8")
        (project / "sft_data" / "train" / "final_v2" / "sft_honest_abstention.json").write_text(
            "[]", encoding="utf-8")

        image = project / "sft_data" / "variants" / "ADM_x_png.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (10, 20, 30)).save(image)
        source = project / "dataset" / "GenImage_Test" / "ADM" / "x.png"
        source.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), (30, 20, 10)).save(source)
        (project / "sft_data" / "variants" / "manifest.json").write_text(json.dumps({
            "variants": {"sft_data/variants/ADM_x_png.png": {
                "variant_id": "ADM_x_png", "source_id": "ADM/x", "label": "Fake",
                "generator": "ADM", "split": "train", "resolution": [32, 32],
                "treatment": "png", "container": "png", "quality": None}}}), encoding="utf-8")
        (project / "sft_data" / "split_v2.json").write_text(json.dumps({
            "hash": "h", "sources": {"ADM/x": {"split": "train",
                                               "path": "dataset/GenImage_Test/ADM/x.png"}}}),
            encoding="utf-8")
        trajectories = project / "sft_data" / "trajectories"
        trajectories.mkdir(parents=True)
        for policy in ("noise", "no-tool"):
            (trajectories / f"{policy}__ADM_x_png.json").write_text(
                json.dumps(_trajectory(policy=policy)), encoding="utf-8")
        return project

    def test_it_writes_pages_sets_and_a_worklist(self, tmp_path, monkeypatch):
        project = self._fixture(tmp_path)
        monkeypatch.setattr("scripts.build_review_workstation.PROJECT_ROOT", str(project),
                            raising=False)
        report = build(
            final_v2_dir=str(project / "sft_data" / "train" / "final_v2"),
            output_dir=str(project / "sft_data" / "review"),
            trajectories_dir=str(project / "sft_data" / "trajectories"),
            variants_manifest=str(project / "sft_data" / "variants" / "manifest.json"),
            split_path=str(project / "sft_data" / "split_v2.json"),
        )
        review = project / "sft_data" / "review"
        assert report["samples"] == 1
        assert (review / "index.html").exists()
        assert (review / "samples" / "f2_noise__ADM_x_png.html").exists()
        assert (review / "review_sets.json").exists()

        with open(review / "worklist.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["sample_id"] == "f2_noise__ADM_x_png"
        assert rows[0]["bucket"] == "positive"
        assert rows[0]["decision"] == ""      # the reviewer fills these in

    def test_the_worklist_marks_who_needs_a_second_reader(self, tmp_path, monkeypatch):
        project = self._fixture(tmp_path)
        monkeypatch.setattr("scripts.build_review_workstation.PROJECT_ROOT", str(project),
                            raising=False)
        build(final_v2_dir=str(project / "sft_data" / "train" / "final_v2"),
              output_dir=str(project / "sft_data" / "review"),
              trajectories_dir=str(project / "sft_data" / "trajectories"),
              variants_manifest=str(project / "sft_data" / "variants" / "manifest.json"),
              split_path=str(project / "sft_data" / "split_v2.json"))
        sets = json.loads((project / "sft_data" / "review" / "review_sets.json")
                          .read_text(encoding="utf-8"))
        assert sets["full"] == ["f2_noise__ADM_x_png"]
        assert "high_confidence" in sets["double_review"]["f2_noise__ADM_x_png"]


class TestLoadSamples:
    def test_it_tags_the_bucket(self, tmp_path):
        (tmp_path / "sft_tool_positive.json").write_text(json.dumps([_sample()]),
                                                         encoding="utf-8")
        (tmp_path / "sft_no_tool_positive.json").write_text(json.dumps([]), encoding="utf-8")
        samples = load_samples(str(tmp_path))
        assert samples[0]["_bucket_file"] == "tool_positive"
