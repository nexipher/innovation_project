"""Tests for the G4-f review workstation (plan.md §4.11, 2026-10-09 revision)."""

import csv
import html as html_lib
import json
import os

import pytest
from PIL import Image

from scripts.build_review_workstation import (
    BUCKET_CHECKS,
    COMMON_CHECKS,
    build,
    checklists_for,
    high_risk_reasons,
    load_samples,
    render_sample_page,
)


def _token(**overrides):
    token = {
        "evidence_id": "E-1", "evidence_name": "noise_residual_inconsistency",
        "source": "noise_expert", "measurement_scope": "global",
        "region_semantics": "diagnostic_evidence_region",
        "raw_metric": 1.5235, "strength": 0.1285,
        "region": "patch_coordinates_[0, 0, 26, 26]",
        "region_pixels": [0, 0, 26, 26],
        "region_normalized_1000": [0, 0, 100, 100],
        "coordinate_space": "pixels",
        "phenomenon": "局部噪声方差塌缩。", "reasoning": "低残差微噪声水平。",
        "counter_explanation": "降噪与重压缩同样会改变该水平。",
        "reliability": 0.845,
        "calibrated_likelihood": {"Real": 0.152, "Fake": 0.848},
        "support": "AI-generated", "support_raw": "camera capture",
        "direction": "AI-generated", "direction_source": "calibrated_likelihood",
        "semantics_aligned": False,
        "applicability": "inverted:high-metric-means-real",
        "applicability_conditions": "高 strength 统计上对应 Real。",
        "interpretation_text": "Calibration: ... P(Fake)=0.85 ...",
        "interpretation_text_raw": "Smooth, low-variance output of the kind ...",
        "condition_metadata": {"bin_samples": 79, "region_area_ratio": 0.010315},
        "diagnostic_region_image": "traces/evidence/s1/region_E-1.png",
        "visual_artifacts": ["traces/evidence/s1/noise_residual_map_E-1.png"],
    }
    token.update(overrides)
    return token


def _sample(sample_id="f2_noise__ADM_x_png", verdict="Fake", confidence=0.9,
            truth="Fake", tools=("noise",), treatment="png", generator="ADM",
            bucket="tool_positive", tokens=None, conversations=None):
    return {
        "id": sample_id,
        "image_path": "sft_data/variants/ADM_x_png.png",
        "ground_truth": truth,
        "source_model": generator,
        "final_verdict": {"verdict": verdict, "confidence": confidence,
                          "posterior": 0.9 if verdict == "Fake" else 0.1,
                          "policy_reasons": ["candidate_matches_posterior"]},
        "conversations": conversations if conversations is not None else [
            {"from": "user", "value": "<image>\n请分析。"},
            {"from": "gpt", "value": "<planning>…</planning>"},
            {"from": "user", "value": '{"evidence_id": "E-1"}',
             "image_paths": ["traces/evidence/s1/region_E-1.png"]},
            {"from": "gpt", "value": "<observation>只有容器是 JPEG。</observation>"},
        ],
        "evidence_chain": tokens if tokens is not None else [_token()],
        "metadata": {"trajectory_id": "noise__ADM_x_png", "policy": "noise",
                     "treatment": treatment, "split": "train", "tools_served": list(tools),
                     "admission_category": "positive", "model_candidate": "Uncertain",
                     "counters": {"weighted_cost": 2.0}},
        "type": "positive",
        "_bucket_file": bucket,
    }


def _trajectory(variant_id="ADM_x_png", policy="noise", conflict=0.0, brier=0.02):
    return {
        "trajectory_id": f"{policy}__{variant_id}", "policy": policy,
        "variant_id": variant_id, "conflict_score": conflict,
        "policy_reasons": ["candidate_matches_posterior"],
        "scores": {"posterior_brier": brier, "model_brier": 0.25},
        "counters": {"weighted_cost": 2.0}, "model_candidate": "Uncertain",
        "model_probability": 0.5, "posterior": 0.9, "halting_reason": "model_stalled",
        "final_verdict": "Fake",
    }


class TestHighRiskFlags:
    """Attention markers, not a second review — the naming is the point."""

    def test_an_uncertain_verdict_is_flagged(self):
        assert "uncertain" in high_risk_reasons(_sample(verdict="Uncertain"), _trajectory())

    def test_multiple_tools_are_flagged(self):
        assert "multi_tool" in high_risk_reasons(
            _sample(tools=("noise", "jpeg")), _trajectory())

    def test_a_confident_label_is_flagged(self):
        assert "high_confidence" in high_risk_reasons(
            _sample(confidence=0.9), _trajectory())

    def test_an_open_conflict_is_flagged(self):
        assert "conflict" in high_risk_reasons(_sample(), _trajectory(conflict=0.8))

    def test_a_low_confidence_single_tool_answer_is_not(self):
        assert high_risk_reasons(_sample(verdict="Fake", confidence=0.6), _trajectory()) == []


class TestChecklists:
    def test_every_bucket_gets_the_common_checks(self):
        for bucket in ("tool_positive", "no_tool_positive", "honest_abstention"):
            names = [name for name, _ in checklists_for(bucket)]
            assert [name for name, _ in COMMON_CHECKS] == names[:len(COMMON_CHECKS)]

    def test_the_tool_bucket_asks_for_a_real_gain(self):
        assert "gain_real" in [name for name, _ in checklists_for("tool_positive")]

    def test_the_no_tool_bucket_asks_about_fabricated_evidence(self):
        names = [name for name, _ in checklists_for("no_tool_positive")]
        assert "no_fabricated_evidence" in names
        assert "no_shorcut_real" in names

    def test_abstentions_are_not_asked_for_a_brier_gain(self):
        """140 samples are admitted *because* they make no directional claim."""
        names = [name for name, _ in checklists_for("honest_abstention")]
        assert "gain_real" not in names
        assert "abstention_reasoned" in names

    def test_every_bucket_specific_check_exists(self):
        for bucket, checks in BUCKET_CHECKS.items():
            assert checks, bucket


class TestSamplePage:
    def _render(self, tmp_path, sample=None, source="", missing=None, high_risk=()):
        page = str(tmp_path / "samples" / "x.html")
        os.makedirs(os.path.dirname(page), exist_ok=True)
        return render_sample_page(sample or _sample(), _trajectory(),
                                  _trajectory(policy="no-tool"), source,
                                  str(tmp_path / "index.html"), page,
                                  missing if missing is not None else set(),
                                  list(high_risk), "a_previous_id", "a_next_id",
                                  total_samples=237)

    def test_it_renders_every_conversation_turn_in_order(self, tmp_path):
        """The regression: only the first assistant turn used to be shown."""
        page = self._render(tmp_path)
        planning = page.index("&lt;planning&gt;")
        evidence = page.index("&quot;evidence_id&quot;: &quot;E-1&quot;")
        answer = page.index("&lt;observation&gt;")
        assert planning < evidence < answer

    def test_it_marks_which_turns_training_supervises(self, tmp_path):
        page = self._render(tmp_path)
        assert page.count("参与训练 · 被监督") == 2      # the call turn and the answer
        assert page.count("掩码 · 不进 loss") == 2       # question and evidence round

    def test_the_conversation_summary_counts_the_turns(self, tmp_path):
        page = self._render(tmp_path)
        assert "共 4 轮：助手 2 轮参与训练，用户/系统 2 轮掩码" in page

    def test_evidence_turns_are_indented_for_reading(self, tmp_path):
        page = self._render(tmp_path)
        assert '{\n &quot;evidence_id&quot;: &quot;E-1&quot;\n}' in page

    def test_the_evidence_card_carries_the_full_semantics(self, tmp_path):
        page = self._render(tmp_path)
        for label in ("evidence_name", "phenomenon", "reasoning", "counter_explanation",
                      "reliability", "semantics_aligned", "direction ← direction_source",
                      "condition_metadata", "region_pixels"):
            assert label in page, label
        assert "noise_residual_inconsistency" in page
        assert "0.845" in page                            # reliability
        assert "0.848" in page                            # P(Fake)
        assert "bin_samples: 79" in page                  # condition metadata
        assert "[0, 0, 26, 26]" in page                   # pixel coordinates
        assert "[0, 0, 100, 100]" in page                 # normalized coordinates

    def test_a_flipped_semantics_token_is_warned_about_in_red(self, tmp_path):
        page = self._render(tmp_path)
        assert "semantics_aligned = false" in page
        assert "warn-red" in page
        assert "原始解释" in page and "反解释" in page

    def test_the_three_direction_fields_are_kept_apart(self, tmp_path):
        page = self._render(tmp_path)
        assert "support（整流后主张）" in page
        assert "support_raw（专家原始主张）" in page
        assert "camera capture" in page                    # the expert's own claim

    def test_an_aligned_token_gets_no_red_warning(self, tmp_path):
        sample = _sample(tokens=[_token(semantics_aligned=True)])
        page = self._render(tmp_path, sample=sample)
        assert "semantics_aligned = false" not in page
        assert "semantics_aligned 缺失" not in page

    def test_the_confidence_block_separates_the_four_numbers(self, tmp_path):
        page = self._render(tmp_path)
        assert "原始模型判定" in page and "p=0.5" in page
        assert "停止后验 P(Fake)" in page and "停止后验 P(Real)" in page
        assert "所选标签置信度" in page
        assert "ΔBrier（模型 → 后验）" in page

    def test_a_tool_free_sample_says_there_is_no_posterior(self, tmp_path):
        sample = _sample(tools=(), bucket="no_tool_positive",
                         conversations=[{"from": "user", "value": "看图"},
                                        {"from": "gpt", "value": "<verdict>{}</verdict>"}])
        sample["final_verdict"]["posterior"] = None      # what the pipeline writes
        page = self._render(tmp_path, sample=sample)
        assert "无工具轨迹没有后验" in page
        assert "ΔBrier（模型 → 后验）" not in page      # the row, not the caveat
        assert "ΔBrier 没有意义" in page

    def test_it_uses_the_bucket_checklist_not_a_global_one(self, tmp_path):
        page = self._render(tmp_path, sample=_sample(bucket="honest_abstention"))
        assert html_lib.escape(BUCKET_CHECKS["honest_abstention"][0][1]) in page
        assert "ΔBrier<0 且结论更正确" not in page

    def test_high_risk_marks_are_shown_as_flags(self, tmp_path):
        page = self._render(tmp_path, high_risk=["uncertain", "multi_tool"])
        assert "高风险标记：uncertain" in page
        assert "高风险标记：multi_tool" in page

    def test_the_page_can_save_decisions_without_a_server(self, tmp_path):
        page = self._render(tmp_path)
        assert 'id="notes"' in page and 'data-decision="accept"' in page
        assert "localStorage" in page and "finalv2review.v1" in page
        assert 'id="go-prev"' in page and 'id="go-next"' in page
        assert "快捷键" in page
        assert 'data-samples="237"' in page

    def test_a_missing_image_is_flagged_and_counted(self, tmp_path):
        missing = set()
        page = self._render(tmp_path, missing=missing)
        assert "文件缺失" in page
        assert len(missing) > 0

    def test_text_cannot_inject_markup(self, tmp_path):
        sample = _sample(conversations=[
            {"from": "user", "value": "看图"},
            {"from": "gpt", "value": "<observation><script>alert(1)</script></observation>"}])
        page = self._render(tmp_path, sample=sample)
        assert "<script>alert(1)</script>" not in page
        assert "&lt;script&gt;" in page

    def test_a_token_without_calibration_is_flagged(self, tmp_path):
        token = _token()
        token.pop("calibrated_likelihood")
        token["semantics_aligned"] = None
        sample = _sample(tokens=[token])
        page = self._render(tmp_path, sample=sample)
        assert "semantics_aligned 缺失" in page


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

    def _build(self, project):
        return build(
            final_v2_dir=str(project / "sft_data" / "train" / "final_v2"),
            output_dir=str(project / "sft_data" / "review"),
            trajectories_dir=str(project / "sft_data" / "trajectories"),
            variants_manifest=str(project / "sft_data" / "variants" / "manifest.json"),
            split_path=str(project / "sft_data" / "split_v2.json"))

    def test_it_writes_pages_sets_and_a_worklist(self, tmp_path, monkeypatch):
        project = self._fixture(tmp_path)
        monkeypatch.setattr("scripts.build_review_workstation.PROJECT_ROOT", str(project),
                            raising=False)
        report = self._build(project)
        review = project / "sft_data" / "review"
        assert report["samples"] == 1
        assert (review / "index.html").exists()
        assert (review / "samples" / "f2_noise__ADM_x_png.html").exists()
        assert (review / "review_sets.json").exists()

        with open(review / "worklist.csv", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert rows[0]["sample_id"] == "f2_noise__ADM_x_png"
        assert rows[0]["bucket"] == "tool_positive"
        assert rows[0]["decision"] == ""      # the reviewer fills these in

    def test_the_sets_mark_high_risk_and_declare_a_single_pass(self, tmp_path, monkeypatch):
        project = self._fixture(tmp_path)
        monkeypatch.setattr("scripts.build_review_workstation.PROJECT_ROOT", str(project),
                            raising=False)
        self._build(project)
        sets = json.loads((project / "sft_data" / "review" / "review_sets.json")
                          .read_text(encoding="utf-8"))
        assert sets["full"] == ["f2_noise__ADM_x_png"]
        assert sets["protocol"] == "single_pass"
        assert "high_confidence" in sets["high_risk"]["f2_noise__ADM_x_png"]
        assert "double_review" not in sets
        assert "stratified" not in sets

    def test_the_index_lets_the_reviewer_save_and_export(self, tmp_path, monkeypatch):
        project = self._fixture(tmp_path)
        monkeypatch.setattr("scripts.build_review_workstation.PROJECT_ROOT", str(project),
                            raising=False)
        self._build(project)
        index = (project / "sft_data" / "review" / "index.html").read_text(encoding="utf-8")
        assert 'id="export"' in index and 'id="import"' in index
        assert 'data-row-sample="f2_noise__ADM_x_png"' in index
        assert 'id="only-pending"' in index


class TestLoadSamples:
    def test_it_tags_the_bucket(self, tmp_path):
        (tmp_path / "sft_tool_positive.json").write_text(json.dumps([_sample()]),
                                                        encoding="utf-8")
        (tmp_path / "sft_no_tool_positive.json").write_text(json.dumps([]), encoding="utf-8")
        samples = load_samples(str(tmp_path))
        assert samples[0]["_bucket_file"] == "tool_positive"
