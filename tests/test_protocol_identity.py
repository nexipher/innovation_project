"""
验收协议身份：归一化、比较、文件读取（plan §4.12b）。

重点是**比较的是实际构建出来的配置**：`runtime_identity()` 从管线对象的
client/experts 采集，`identity_from_metadata()` 从 b2 批次元数据采集，两者经过
同一个 `canonical_identity()`。任何一项变了都必须被指出到字段。
"""

import copy
import json
import os

import pytest

from utils.protocol_identity import (REQUIRED_SECTIONS, SCHEMA, assert_compatible,
                                     canonical_identity, compatibility_problems,
                                     identity_from_metadata, identity_sha256,
                                     load_accepted, pack_identity,
                                     protocol_file_paths, runtime_identity)

pytestmark = pytest.mark.core


def _base_identity():
    return canonical_identity(
        model={"name": "m", "checkpoint": "ckpt", "revision": "rev",
               "dtype": "bfloat16", "inference_mode": "non-thinking",
               "generation": {"temperature": 0.0, "do_sample": False,
                              "max_new_tokens": 1024}},
        prompts={"digest": "digest", "forensic_sha256": "prompt-hash"},
        experts={"probe_dino_expert": {
            "class": "ProbeDinoExpert", "source_name": "probe_dino_expert",
            "weights_sha256": "weights", "weights_bytes": 123,
            "preprocess": {"sha256": "prep", "detail": "整图"}}},
        tools={"policy": "g8_default", "policy_version": "v3",
               "tools_served": ["probe_dino_expert"],
               "evidence_injection": "text+image"},
        budget={"halting_policy": "v3", "max_expert_calls": 5,
                "max_model_turns": 6, "evidence_injection": "text+image"},
        decoding={"greedy": True, "inference_mode": "non-thinking"},
        provenance={"files": {"state_machine/controller.py": "abc"},
                    "sha256": "prov"})


class TestComparison:
    def test_identical_identities_have_no_problems(self):
        assert compatibility_problems(_base_identity(), _base_identity()) == []

    @pytest.mark.parametrize("mutate,expected", [
        (lambda value: value["model"].__setitem__("revision", "other"),
         "model.revision"),
        (lambda value: value["model"].__setitem__("dtype", "float16"),
         "model.dtype"),
        (lambda value: value["model"]["generation"].__setitem__(
            "temperature", 0.1), "model.generation.temperature"),
        (lambda value: value["prompts"].__setitem__("forensic_sha256", "x"),
         "prompts.forensic_sha256"),
        (lambda value: value["experts"]["probe_dino_expert"].__setitem__(
            "weights_sha256", "x"), "experts.probe_dino_expert.weights_sha256"),
        (lambda value: value["experts"]["probe_dino_expert"].__setitem__(
            "preprocess_sha256", "x"),
         "experts.probe_dino_expert.preprocess_sha256"),
        (lambda value: value["evidence_pack"].__setitem__("version", "other"),
         "evidence_pack.version"),
        (lambda value: value["tools"].__setitem__(
            "tools_served", ["probe_dino_expert", "npr_expert"]),
         "tools.tools_served"),
        (lambda value: value["budget"].__setitem__("max_expert_calls", 99),
         "budget.max_expert_calls"),
        (lambda value: value["decoding"].__setitem__("greedy", False),
         "decoding.greedy"),
        (lambda value: value["provenance"]["files"].__setitem__(
            "state_machine/controller.py", "changed"),
         "provenance.files.state_machine/controller.py"),
    ])
    def test_every_protocol_field_is_compared(self, mutate, expected):
        actual = _base_identity()
        mutate(actual)
        problems = compatibility_problems(_base_identity(), actual)
        assert any(expected in problem for problem in problems), problems

    def test_a_missing_or_extra_expert_is_reported_by_name(self):
        actual = _base_identity()
        actual["experts"]["npr_expert"] = {
            "class": "NPRExpert", "source_name": "npr_expert",
            "weights_sha256": "w", "weights_bytes": 1,
            "preprocess_sha256": "p", "preprocess_detail": "d"}
        problems = compatibility_problems(_base_identity(), actual)
        assert any("experts.npr_expert" in problem for problem in problems)

    def test_an_incomplete_accepted_identity_is_refused(self):
        accepted = _base_identity()
        del accepted["budget"]
        problems = compatibility_problems(accepted, _base_identity())
        assert problems and "budget" in problems[0]

    def test_assert_compatible_names_the_field_and_values(self):
        accepted = {"identity": _base_identity()}
        actual = _base_identity()
        actual["model"]["revision"] = "different"
        with pytest.raises(SystemExit) as caught:
            assert_compatible(accepted, actual, generation="g8", batch="b2")
        message = str(caught.value)
        assert "model.revision" in message
        assert "验收 'rev'" in message and "当前 'different'" in message

    def test_assert_compatible_accepts_identical_identities(self):
        assert_compatible({"identity": _base_identity()}, _base_identity(),
                          generation="g8", batch="b2")


class TestTheSharedShape:
    def test_pack_version_is_derived_from_the_key_set(self):
        from state_machine.evidence_tokenizer import EvidenceTokenizer

        identity = pack_identity()
        assert identity["version"].startswith("slim/")
        assert identity["keys"] == list(EvidenceTokenizer.MODEL_VISIBLE_KEYS)

    def test_metadata_and_runtime_identities_compare_equal(self, tmp_path):
        """两侧走同一个归一化：b2 元数据身份 == 实际管线对象身份。"""
        import types

        weights = tmp_path / "weights.pth"
        weights.write_bytes(b"fake weights")

        class StubClient:
            name = "stub"

            def identity(self):
                return {"name": "Qwen3-VL-8B-Instruct", "checkpoint": "ckpt",
                        "revision": "rev", "dtype": "bfloat16",
                        "inference_mode": "non-thinking",
                        "licence": "apache-2.0",
                        "snapshot": "/machine/specific/path",
                        "generation": {"temperature": 0.0, "do_sample": False,
                                       "max_new_tokens": 1024}}

        class ProbeDinoExpert:        # 名字要与 EXPERT_FILES 的登记对上
            source_name = "probe_dino_expert"
            metric_polarity = 1
            device = "cuda"

            def __init__(self, path):
                self.weights_path = path

            def identity(self):
                return {"expert": "probe_dino_expert"}

        base = types.SimpleNamespace(
            _mllm=StubClient(),
            _experts={"probe_dino_expert": ProbeDinoExpert(str(weights))})
        actual = runtime_identity(base, tools_policy="g8_default",
                                  greedy=True, inference_mode="non-thinking")

        metadata = {
            "generation": "g8", "batch": "b2",
            "policy": "g8_default", "policy_version": "v3",
            "tools_served": ["probe_dino_expert"],
            "evidence_injection": "text+image",
            "decoding": {"greedy": True, "inference_mode": "non-thinking"},
            "model": StubClient().identity(),
            "prompts": {"digest": actual["prompts"]["digest"],
                        "forensic_sha256": actual["prompts"]["forensic_sha256"]},
            "experts": {"probe_dino_expert": {
                "class": "ProbeDinoExpert",
                "source_name": "probe_dino_expert",
                "weights_sha256": actual["experts"]["probe_dino_expert"][
                    "weights_sha256"],
                "weights_bytes": weights.stat().st_size}},
            "budget": actual["budget"],
        }
        accepted = identity_from_metadata(metadata)
        assert compatibility_problems(accepted, actual) == []

    def test_an_unknown_expert_class_cannot_be_bound(self):
        with pytest.raises(SystemExit) as caught:
            protocol_file_paths(["MysteryExpert"])
        assert "MysteryExpert" in str(caught.value)

    def test_identity_hash_is_stable_across_dict_order(self):
        first = _base_identity()
        second = json.loads(json.dumps(first))
        assert identity_sha256(first) == identity_sha256(second)


class TestLoadingTheAcceptedFile:
    def _write(self, tmp_path, payload):
        path = tmp_path / "accepted.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return str(path)

    def test_a_missing_file_is_refused(self, tmp_path):
        with pytest.raises(SystemExit) as caught:
            load_accepted(str(tmp_path / "missing.json"))
        assert "尚未登记" in str(caught.value)

    def test_a_broken_file_is_refused(self, tmp_path):
        path = tmp_path / "broken.json"
        path.write_text("{broken", encoding="utf-8")
        with pytest.raises(SystemExit) as caught:
            load_accepted(str(path))
        assert "无法读取" in str(caught.value)

    def test_a_wrong_schema_is_refused(self, tmp_path):
        path = self._write(tmp_path, {"schema": "other/1", "identity": {}})
        with pytest.raises(SystemExit) as caught:
            load_accepted(path)
        assert "schema" in str(caught.value)

    def test_a_missing_section_is_refused(self, tmp_path):
        identity = _base_identity()
        identity.pop("prompts")
        path = self._write(tmp_path, {"schema": SCHEMA, "identity": identity})
        with pytest.raises(SystemExit) as caught:
            load_accepted(path)
        assert "prompts" in str(caught.value)

    def test_a_tampered_identity_is_refused(self, tmp_path):
        identity = _base_identity()
        path = self._write(tmp_path, {"schema": SCHEMA, "identity": identity,
                                      "identity_sha256": "not-the-hash"})
        with pytest.raises(SystemExit) as caught:
            load_accepted(path)
        assert "identity_sha256" in str(caught.value)

    def test_a_missing_hash_is_refused(self, tmp_path):
        path = self._write(tmp_path, {"schema": SCHEMA,
                                      "identity": _base_identity()})
        with pytest.raises(SystemExit) as caught:
            load_accepted(path)
        assert "identity_sha256" in str(caught.value)

    def test_a_complete_file_loads(self, tmp_path):
        identity = _base_identity()
        path = self._write(tmp_path, {"schema": SCHEMA, "identity": identity,
                                      "identity_sha256": identity_sha256(identity)})
        payload = load_accepted(path)
        assert payload["identity"]["model"]["revision"] == "rev"
        assert all(section in payload["identity"] for section in REQUIRED_SECTIONS)
