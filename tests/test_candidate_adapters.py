"""
Offline candidate adapters (SPAI / AIDE) — CPU-side contracts.

The heavy model loads (meta key coverage, official transforms) live in
`scripts/check_candidate_weights.py` and are run in the isolated candidate
venv because the host cgroup is 2 GB.  These tests pin the parts that must
hold in the main environment: manifests, direction/threshold, and the expert
boundary around a stubbed evaluator.
"""

import json
import os

import numpy as np
import pytest

from config import PROJECT_ROOT
from experts.candidate_common import load_candidate_code, load_candidate_weights
from experts.spai import SPAIExpert
from experts.aide import AIDEExpert
from scripts.g8_external_eval import expert_orientation

pytestmark = pytest.mark.core


class TestManifests:
    def test_the_weight_manifest_names_four_files_with_hashes(self):
        payload = load_candidate_weights()
        assert payload["schema"] == "candidate_weights/1"
        spai = payload["candidates"]["spai"]
        assert len(spai["sha256"]) == 64 and spai["bytes"] > 0
        aide = payload["candidates"]["aide"]
        assert aide["backbone_included"] is True
        assert set(aide["checkpoints"]) == {"progan_train", "sd14_train",
                                            "GenImage_train"}
        assert aide["checkpoints"]["progan_train"]["primary"] is True
        for entry in aide["checkpoints"].values():
            assert len(entry["sha256"]) == 64 and entry["bytes"] > 0

    def test_the_files_exist_with_the_recorded_sizes(self):
        payload = load_candidate_weights()
        paths = [payload["candidates"]["spai"]]
        paths += list(payload["candidates"]["aide"]["checkpoints"].values())
        for entry in paths:
            if not os.path.exists(entry["path"]):
                pytest.skip(f"candidate weight not uploaded here: {entry['path']}")
            assert os.path.getsize(entry["path"]) == entry["bytes"]

    def test_the_code_manifest_pins_exact_commits(self):
        payload = load_candidate_code()
        for name in ("spai", "aide"):
            entry = payload["pinned"][name]
            assert len(entry["commit"]) == 40
            assert entry["file_count"] > 0
            assert len(entry["source_manifest_sha256"]) == 16


class TestTheImportStubs:
    def test_repeated_find_spec_after_stubbing_does_not_raise(self):
        """The seaborn/pandas stubs must carry a ModuleSpec (regression).

        The stubs are process-global; restore sys.modules and the timm shim
        afterwards so pandas/narwhals users in later tests see the real
        packages (test pollution found in the full core run).
        """
        import sys

        import timm.data.transforms as timm_transforms

        from experts.candidate_common import has_module
        from experts.spai import _install_import_shims

        names = ("seaborn", "pandas", "clip")
        saved = {name: sys.modules.get(name) for name in names}
        had_pil_interp = hasattr(timm_transforms, "_pil_interp")
        try:
            _install_import_shims()
            for _ in range(3):
                assert isinstance(has_module("seaborn"), bool)
                assert isinstance(has_module("pandas"), bool)
        finally:
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module
            if not had_pil_interp and hasattr(timm_transforms, "_pil_interp"):
                delattr(timm_transforms, "_pil_interp")


class TestSemantics:
    @pytest.mark.parametrize("name", ["spai", "aide"])
    def test_candidates_are_higher_is_fake_at_the_official_threshold(self, name):
        field, threshold, direction = expert_orientation(name)
        assert field == "raw_metric"
        assert threshold == pytest.approx(0.5)
        assert direction == "higher_is_fake"

    def test_the_candidate_semantics_file_is_not_the_agent_registry(self):
        path = os.path.join(PROJECT_ROOT, "experts", "candidate_semantics.json")
        payload = json.load(open(path, encoding="utf-8"))
        assert set(payload["experts"]) == {"spai", "aide"}
        # The Agent registry must not have learned about the candidates.
        registry = json.load(open(os.path.join(
            PROJECT_ROOT, "experts", "score_semantics.json"), encoding="utf-8"))
        assert "spai_expert" not in registry["experts"]
        assert "aide_expert" not in registry["experts"]


class _StubEvaluator:
    def __init__(self, score):
        self._score = score

    def score(self, _image):
        return self._score


class TestTheExpertBoundary:
    def _patch(self, expert, score):
        expert._evaluator = _StubEvaluator(score)
        expert._missing_dependency = lambda: ""
        return expert

    @pytest.mark.parametrize("factory", [SPAIExpert, AIDEExpert])
    def test_a_stub_score_becomes_an_uncalibrated_token(self, factory, tmp_path):
        weights = tmp_path / "weights.pth"
        weights.write_bytes(b"not-a-real-checkpoint")
        expert = self._patch(factory(weights_path=str(weights), device="cpu"), 0.75)
        result = expert.analyze(np.zeros((16, 16, 3), dtype=np.uint8))
        assert result.raw_metric == pytest.approx(0.75)
        assert result.support == "Uncertain"
        assert result.region == "full_image"
        assert result.strength == pytest.approx(0.75)

    @pytest.mark.parametrize("factory", [SPAIExpert, AIDEExpert])
    def test_a_missing_weight_file_is_refused(self, factory, tmp_path):
        expert = factory(weights_path=str(tmp_path / "missing.pth"), device="cpu")
        assert "不存在" in expert.unavailable_reason()

    def test_aide_default_checkpoint_is_progan(self, tmp_path):
        weights = tmp_path / "progan_train.pth"
        weights.write_bytes(b"x")
        expert = AIDEExpert(weights_path=str(weights), device="cpu")
        assert expert.checkpoint == "progan_train"
