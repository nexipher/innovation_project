"""
The 200-input external evaluation set (stage 4.10).

100 fixed source images × 2 conditions.  The point of these tests is not the
images themselves but the **claims the report will rest on**: both conditions
exist for every source, the originals are untouched, the JPEG condition used one
encoder with recorded parameters, and nothing was resized.  They skip when the
external set is not on this machine.
"""

import json
import os

import pytest
from PIL import Image

from config import PROJECT_ROOT

pytestmark = pytest.mark.core

BASE = os.path.join(PROJECT_ROOT, "dataset", "mnw_div2k")
PLAN = os.path.join(BASE, "eval_plan.json")
MANIFEST = os.path.join(BASE, "inputs_manifest.json")

needs_data = pytest.mark.skipif(
    not (os.path.exists(PLAN) and os.path.exists(MANIFEST)),
    reason="外部评测数据不在这台机器上")


def _load(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@needs_data
class TestTheFixedInputSet:
    def test_every_source_has_both_conditions(self):
        manifest = _load(MANIFEST)
        conditions = {}
        for entry in manifest["inputs"]:
            conditions.setdefault(entry["source_path"], set()).add(entry["condition"])
        assert len(conditions) == 100, "应当正好 100 个源"
        assert all(found == {"original", "jpeg_q95"} for found in conditions.values())
        assert manifest["counts"] == {"total": 200, "original": 100, "jpeg_q95": 100}

    def test_the_originals_are_the_ones_the_plan_fixed(self):
        plan = _load(PLAN)
        manifest = _load(MANIFEST)
        fixed = {sample["path"]: sample["sha256"]
                 for group in plan["groups"].values() for sample in group}
        originals = [e for e in manifest["inputs"] if e["condition"] == "original"]
        assert len(originals) == len(fixed) == 100
        for entry in originals:
            assert entry["sha256"] == fixed[entry["path"]]
            assert entry["source_sha256"] == entry["sha256"]

    def test_the_jpeg_condition_records_one_encoder_for_real_and_fake(self):
        manifest = _load(MANIFEST)
        jpeg = [e for e in manifest["inputs"] if e["condition"] == "jpeg_q95"]
        assert len(jpeg) == 100
        encoders = {json.dumps(e["encoder"], sort_keys=True) for e in jpeg}
        assert len(encoders) == 1, "真假两半必须是同一编码器同一参数"
        encoder = jpeg[0]["encoder"]
        assert encoder["params"] == {"quality": 95, "subsampling": 2,
                                     "optimize": False, "progressive": False}
        assert encoder["library"] and encoder["jpeg_backend"] != "unknown"

    def test_no_input_was_resized(self):
        manifest = _load(MANIFEST)
        for entry in manifest["inputs"]:
            if entry["condition"] != "jpeg_q95":
                continue
            source = os.path.join(BASE, entry["source_path"])
            target = os.path.join(PROJECT_ROOT, entry["path"])
            with Image.open(source) as original, Image.open(target) as recoded:
                assert recoded.size == original.size == (entry["width"], entry["height"])

    def test_the_manifest_says_what_it_may_not_be_used_for(self):
        manifest = _load(MANIFEST)
        assert manifest["rules"]["no_threshold_fitting"] is True
        assert manifest["rules"]["same_encoder_for_real_and_fake"] is True
        assert "未压缩" in manifest["rules"]["note"]      # 明确禁止 JPEG→PNG 的解释

    def test_the_agent_plan_covers_all_two_hundred_inputs(self):
        plan = _load(PLAN)
        assert plan["agent_plan"]["sessions"] == 200
        assert plan["analysis"]["counts"] == {"inputs": 200, "expert_measurements": 600,
                                              "agent_sessions": 200}
        assert plan["analysis"]["pairing"]                 # 配对比较写明了
        assert "agent_subset" not in plan                  # 50 张子集已取消
