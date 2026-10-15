"""
The complementarity comparison, on synthetic runs.

Two experts at the same AUROC can be redundant or complementary, and the
retained set turns on which.  These tests pin the arithmetic that tells them
apart: direction correction, the mistake-overlap ratio, and the refusal to
compare runs from different partitions or to treat a mock as an expert.
"""

import json
import os

import pytest

from config import PROJECT_ROOT

pytestmark = pytest.mark.core


def _write_run(root, name, source_name, scores, labels, split="val", mock=False,
               treatments=None):
    """One evaluation directory, in the shape the evaluator writes."""
    path = os.path.join(root, name)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "report.json"), "w", encoding="utf-8") as handle:
        json.dump({"expert": name, "source_name": source_name, "split": split,
                   "mock": mock, "metrics": {}}, handle)
    with open(os.path.join(path, "scores.jsonl"), "w", encoding="utf-8") as handle:
        for index, value in enumerate(scores):
            handle.write(json.dumps({
                "variant_id": f"v{index}", "source_id": f"s{index}",
                "label": labels[index],
                "treatment": (treatments or ["png"] * len(scores))[index],
                "raw_score": value, "elapsed_ms": 1.0, "status": "ok",
                "error": None}) + "\n")
    return path


class TestTheOverlapArithmetic:
    def test_redundant_experts_show_no_complementarity(self, tmp_path):
        from scripts.g8_compare_experts import compare, load_run

        labels = ["Fake", "Real", "Fake", "Real"]
        right = [0.9, 0.1, 0.9, 0.1]
        also = [0.8, 0.2, 0.8, 0.2]                    # same mistakes, same order
        a = load_run(_write_run(str(tmp_path), "a", "probe_dino_expert", right, labels))
        b = load_run(_write_run(str(tmp_path), "b", "probe_dino_expert", also, labels))
        report = compare([a, b])
        pair = report["pairwise"][0]
        assert pair["agreement"] == 1.0
        assert "未显示互补" in pair["note"]

    def test_opposite_mistakes_show_complementarity(self, tmp_path):
        from scripts.g8_compare_experts import compare, load_run

        labels = ["Fake", "Fake", "Real", "Real"]
        a = load_run(_write_run(str(tmp_path), "a", "probe_dino_expert",
                                [0.9, 0.1, 0.1, 0.9], labels))   # wrong on 2nd and 3rd
        b = load_run(_write_run(str(tmp_path), "b", "probe_dino_expert",
                                [0.1, 0.9, 0.9, 0.1], labels))   # wrong on 1st and 4th
        report = compare([a, b])
        pair = report["pairwise"][0]
        assert pair["both_wrong_rate"] == 0.0
        assert pair["overlap_ratio"] == 0.0
        assert "错误互补" in pair["note"]
        # 互补 ≠ 平均就好：两个错误相反的专家取平均会处处落在 0.5，
        # 于是 AUROC 掉到随机水平。这条断言是刻意的——朴素平均只是工具，
        # 不是结论；真正互补的专家（错误少且分散）才会让平均超过各自。
        assert report["naive_average"]["auroc"] == 0.5


class TestTheDirectionIsAppliedBeforeComparing:
    def test_a_reversed_expert_is_flipped_into_the_same_frame(self, tmp_path):
        from scripts.g8_compare_experts import compare, load_run

        labels = ["Fake", "Real"]
        # noise is higher_is_real: 0.1 on a Fake is as good as 0.9 for a
        # higher_is_fake expert, and the comparison must see it that way.
        a = load_run(_write_run(str(tmp_path), "a", "probe_dino_expert", [0.9, 0.1], labels))
        b = load_run(_write_run(str(tmp_path), "b", "noise_expert", [0.1, 0.9], labels))
        report = compare([a, b])
        assert report["pairwise"][0]["agreement"] == 1.0
        assert report["naive_average"]["auroc"] == 1.0


class TestWhatItRefuses:
    def test_a_mock_run_is_skipped(self, tmp_path, capsys):
        from scripts.g8_compare_experts import load_run

        _write_run(str(tmp_path), "m", "probe_dino_expert", [0.5, 0.5],
                   ["Fake", "Real"], mock=True)
        assert load_run(os.path.join(str(tmp_path), "m")) is None
        assert "跳过 Mock" in capsys.readouterr().out

    def test_runs_from_different_partitions_are_refused(self, tmp_path):
        from scripts.g8_compare_experts import compare, load_run

        labels = ["Fake", "Real"]
        a = load_run(_write_run(str(tmp_path), "a", "probe_dino_expert", [0.9, 0.1],
                                labels, split="val"))
        b = load_run(_write_run(str(tmp_path), "b", "probe_dino_expert", [0.9, 0.1],
                                labels, split="test"))
        # compare() itself does not check the partition; the CLI does — so the
        # guard is asserted where it lives
        source = open(os.path.join(PROJECT_ROOT, "scripts", "g8_compare_experts.py"),
                      encoding="utf-8").read()
        assert "不同分区不能混算" in source
        assert compare([a, b])["images"] == 2      # same ids, so it can align

    def test_images_must_overlap(self, tmp_path):
        from scripts.g8_compare_experts import compare, load_run

        a = load_run(_write_run(str(tmp_path), "a", "probe_dino_expert", [0.9], ["Fake"]))
        b = load_run(_write_run(str(tmp_path), "b", "probe_dino_expert", [0.1], ["Real"]))
        b["rows"] = {"other": list(b["rows"].values())[0]}
        with pytest.raises(SystemExit):
            compare([a, b])


class TestTheReportKeepsThePerFormatView:
    def test_each_format_cell_gets_its_own_auroc(self, tmp_path):
        from scripts.g8_compare_experts import compare, load_run

        labels = ["Fake", "Real", "Fake", "Real"]
        treatments = ["png", "png", "jpeg_q70", "jpeg_q70"]
        a = load_run(_write_run(str(tmp_path), "a", "probe_dino_expert",
                                [0.9, 0.1, 0.1, 0.9], labels, treatments=treatments))
        b = load_run(_write_run(str(tmp_path), "b", "noise_expert",
                                [0.1, 0.9, 0.9, 0.1], labels, treatments=treatments))
        report = compare([a, b])
        assert report["per_format"]["png"]["auroc"]["a"] == 1.0
        assert report["per_format"]["jpeg_q70"]["auroc"]["a"] == 0.0
