"""
外部评测的统计与报告（阶段 4.11，CPU 部分）。

GPU 上的 600 次测量可以等，但"这些数怎么算"必须先钉住：单标签组算不出 AUROC、
两种条件必须按源配对、判对率只能用登记表既有阈值。这些都在纯函数里，CPU 可测。
"""

import json

import pytest

from scripts.g8_external_eval import (auroc, bootstrap_auroc, build_report,
                                      expert_orientation, oriented, paired_delta,
                                      predicts_fake)

pytestmark = pytest.mark.core


class TestAuroc:
    def test_perfect_separation_is_one(self):
        assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == pytest.approx(1.0)

    def test_reversed_scores_are_zero(self):
        assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == pytest.approx(0.0)

    def test_ties_are_half(self):
        assert auroc([0.5, 0.5], [1, 0]) == pytest.approx(0.5)

    def test_a_single_class_is_not_an_auroc(self):
        """单标签组（例如单个生成器组）算不出 AUROC——必须返回 None，不许猜。"""
        assert auroc([0.9, 0.8], [1, 1]) is None
        assert auroc([0.1, 0.2], [0, 0]) is None


class TestBootstrap:
    def test_it_is_deterministic_given_the_seed(self):
        scores = [0.9, 0.7, 0.6, 0.55, 0.45, 0.4, 0.3, 0.1]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        first = bootstrap_auroc(scores, labels, samples=200, seed=7)
        second = bootstrap_auroc(scores, labels, samples=200, seed=7)
        assert first == second
        assert first["auroc"] == pytest.approx(auroc(scores, labels))
        assert first["ci95"][0] <= first["auroc"] <= first["ci95"][1]

    def test_single_class_returns_none(self):
        assert bootstrap_auroc([0.9, 0.8], [1, 1]) is None


class TestPairedDelta:
    def test_a_condition_that_improves_ordering_gives_a_positive_delta(self):
        # 基线有重叠（AUROC < 1），另一条件完全分开（AUROC = 1）→ Δ > 0
        base = [0.9, 0.4, 0.6, 0.35, 0.5, 0.3, 0.2, 0.1]
        better = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        result = paired_delta(base, better, labels, samples=200, seed=7)
        assert result["delta"] > 0
        assert result["ci95"][0] <= result["delta"] <= result["ci95"][1]

    def test_it_is_paired_not_independent(self):
        """同一条目在两种条件下成对重采样：打乱配对会改变结果。"""
        base = [0.9, 0.8, 0.2, 0.1]
        other = [0.7, 0.6, 0.3, 0.2]
        labels = [1, 1, 0, 0]
        paired = paired_delta(base, other, labels, samples=200, seed=7)
        shuffled = paired_delta(base, list(reversed(other)), labels, samples=200, seed=7)
        assert paired != shuffled


class TestTheRegistryDecidesDirectionAndThreshold:
    def test_probe_is_higher_is_fake(self):
        field, threshold, direction = expert_orientation("probe")
        assert (field, direction) == ("raw_metric", "higher_is_fake")
        assert predicts_fake(0.9, direction, threshold) is True
        assert predicts_fake(0.1, direction, threshold) is False

    def test_a_higher_is_real_expert_flips_both(self):
        field, threshold, direction = expert_orientation("npr")
        assert direction == "higher_is_fake"          # NPR-A 已选定为 higher_is_fake
        assert predicts_fake(0.9, direction, threshold) is True

    def test_orientation_flips_the_score_for_higher_is_real(self):
        assert oriented(0.9, "higher_is_fake") == pytest.approx(0.9)
        assert oriented(0.9, "higher_is_real") == pytest.approx(-0.9)
        assert predicts_fake(0.1, "higher_is_real", 0.5) is True
        assert predicts_fake(0.9, "higher_is_real", 0.5) is False


def _row(variant_id, group, condition, label, expert, value,
         direction="higher_is_fake", status="measured", error=""):
    # 配对键是源：同一源在两种条件下 source_path 必须相同（与真实清单一致）
    return {"variant_id": variant_id, "expert": expert, "group": group,
            "condition": condition, "label": label, "source_id": variant_id,
            "source_path": variant_id,
            "field": "raw_metric", "value": value, "direction": direction,
            "oriented": (oriented(value, direction) if value is not None else None),
            "status": status, "error": error, "threshold": 0.5,
            "elapsed_ms": 1.0}


class TestTheReport:
    def _rows(self):
        """生成器组只有 Fake；Real 来自独立的 DIV2K 组——与真实设计一致。"""
        rows = []
        for index in range(2):
            fake = f"gen__f{index}"
            real = f"div2k__r{index}"
            for condition, bump in (("original", 0.0), ("jpeg_q95", 0.1)):
                rows.append(_row(fake, "gen", condition, "Fake", "probe", 0.7 + bump))
                rows.append(_row(real, "div2k", condition, "Real", "probe", 0.3 + bump))
        return rows

    def test_the_source_group_view_has_no_auroc(self):
        """来源组是单标签：只给均值与阈值判假率，AUROC 必须留给配对视图。"""
        report = build_report(self._rows(), ["probe"], {"inputs": []}, "m.json")
        original = report["groups"]["gen"]["original"]
        assert (original["n"], original["n_fake"], original["n_real"]) == (2, 2, 0)
        data = original["per_expert"]["probe"]
        assert data["mean_score"] == pytest.approx(0.7)
        assert data["fake_rate_at_threshold"] == pytest.approx(1.0)
        assert "auroc" not in data

    def test_the_paired_view_pairs_generator_fakes_with_div2k_reals(self):
        report = build_report(self._rows(), ["probe"], {"inputs": []}, "m.json")
        bucket = report["analysis"]["gen vs div2k"]["original"]
        assert (bucket["n"], bucket["n_fake"], bucket["n_real"]) == (4, 2, 2)
        assert bucket["per_expert"]["probe"]["auroc"]["auroc"] == pytest.approx(1.0)

    def test_the_paired_delta_is_attached_to_the_original_condition(self):
        report = build_report(self._rows(), ["probe"], {"inputs": []}, "m.json")
        delta = report["analysis"]["gen vs div2k"]["original"]["per_expert"]["probe"][
            "paired_delta_vs_jpeg_q95"]
        assert delta is not None and "delta" in delta

    def test_the_same_real_group_is_reused_for_every_generator(self):
        rows = self._rows()
        for index in range(2):
            rows.append(_row(f"gen2__f{index}", "gen2", "original", "Fake", "probe", 0.8))
            rows.append(_row(f"gen2__f{index}", "gen2", "jpeg_q95", "Fake", "probe", 0.8))
        report = build_report(rows, ["probe"], {"inputs": []}, "m.json")
        assert report["analysis"]["gen vs div2k"]["original"]["n_real"] == 2
        assert report["analysis"]["gen2 vs div2k"]["original"]["n_real"] == 2

    def test_the_report_says_what_it_may_not_be_used_for(self):
        report = build_report(self._rows(), ["probe"], {"inputs": []}, "m.json")
        assert report["rules"]["no_threshold_fitting"] is True
        assert report["rules"]["paired_conditions"] is True
        assert report["rules"]["real_images_reused_across_generators"] is True

    def test_the_report_records_the_label_container_confound(self):
        """original 条件里 Fake=JPEG / Real=PNG；报告必须自己写出来。"""
        manifest = {"inputs": [
            {"condition": "original", "label": "Fake", "container": "jpg", "path": "a.jpg"},
            {"condition": "original", "label": "Real", "container": "png", "path": "b.png"},
            {"condition": "jpeg_q95", "label": "Fake", "container": "jpg", "path": "c.jpg"},
            {"condition": "jpeg_q95", "label": "Real", "container": "jpg", "path": "d.jpg"},
        ]}
        report = build_report(self._rows(), ["probe"], manifest, "m.json")
        assert report["composition"]["original"]["by_label_container"] == {
            "Fake/jpg": 1, "Real/png": 1}
        assert report["composition"]["jpeg_q95"]["by_label_container"] == {
            "Fake/jpg": 1, "Real/jpg": 1}

    def test_each_expert_is_reported_with_weights_and_preprocessing(self):
        report = build_report(self._rows(), ["probe"], {"inputs": []}, "m.json")
        identity = report["experts"]["probe"]
        assert identity["weights"]["sha256"]
        assert identity["preprocess"]["measurement_scope"] == "global"
        assert identity["threshold"]["value"] == pytest.approx(0.5)

    def test_a_failed_measurement_is_counted_not_scored_as_zero(self):
        rows = self._rows()
        failed = rows[0]
        rows[0] = {**failed, "value": None, "oriented": None,
                   "status": "failed", "error": "cuda OOM"}
        report = build_report(rows, ["probe"], {"inputs": []}, "m.json")
        bucket = report["analysis"]["gen vs div2k"]["original"]
        data = bucket["per_expert"]["probe"]
        assert data["failures"] == 1
        assert data["n_measured"] == 3
        assert data["n_fake_measured"] == 1
        assert data["auroc"] is not None
        # The failed value never enters the mean or the fake-rate denominator.
        assert data["n_measured"] + data["failures"] == bucket["n"]

    def test_all_failures_produce_no_auroc_and_no_zero_score(self):
        rows = [{**row, "value": None, "oriented": None,
                 "status": "failed", "error": "missing weights"}
                for row in self._rows()]
        report = build_report(rows, ["probe"], {"inputs": []}, "m.json")
        data = report["analysis"]["gen vs div2k"]["original"]["per_expert"]["probe"]
        assert data["auroc"] is None
        assert data["threshold_accuracy"] is None
        assert data["balanced_accuracy"] is None
        assert data["n_measured"] == 0
        assert data["failures"] == 4

    def test_tpr_tnr_and_balanced_accuracy(self):
        rows = [
            _row("f0", "gen", "original", "Fake", "probe", 0.9),
            _row("f1", "gen", "original", "Fake", "probe", 0.4),
            _row("r0", "div2k", "original", "Real", "probe", 0.6),
            _row("r1", "div2k", "original", "Real", "probe", 0.2),
        ]
        report = build_report(rows, ["probe"], {"inputs": []}, "m.json")
        data = report["analysis"]["gen vs div2k"]["original"]["per_expert"]["probe"]
        assert data["tpr"] == pytest.approx(0.5)
        assert data["tnr"] == pytest.approx(0.5)
        assert data["balanced_accuracy"] == pytest.approx(0.5)
        assert data["elapsed_ms"]["n"] == 4
        assert data["elapsed_ms"]["median"] == pytest.approx(1.0)
        assert report["analysis"]["gen vs div2k"]["original"]["n_sources"] == 4

    def test_a_single_label_pair_has_no_auroc(self):
        rows = [row for row in self._rows() if row["label"] == "Fake"]
        report = build_report(rows, ["probe"], {"inputs": []}, "m.json")
        bucket = report["analysis"]["gen vs None"]["original"]
        assert bucket["per_expert"]["probe"]["auroc"] is None
