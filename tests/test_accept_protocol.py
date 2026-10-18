"""
验收协议身份的登记（plan §4.12b）。

登记只在**逐样本验收全部通过**后发生，并且必须证明协议文件自 b2 运行提交后
没有变化；否则 b2 的人工验收不覆盖当前协议。测试用 tmp 批次，不写仓库。
"""

import json
import os

import pytest

from scripts import g8_accept_protocol as accept
from tests.test_batch_gate import BATCH, GENERATION, POLICY, Batch, _decisions, _ledger

pytestmark = pytest.mark.core


def _metadata(batch):
    return {
        "generation": GENERATION, "batch": BATCH, "policy": POLICY,
        "policy_version": "v3", "tools_served": ["probe_dino_expert"],
        "evidence_injection": "text+image",
        "decoding": {"greedy": True, "inference_mode": "non-thinking"},
        "model": {"name": "Qwen3-VL-8B-Instruct", "checkpoint": "ckpt",
                  "revision": "rev", "dtype": "bfloat16",
                  "inference_mode": "non-thinking",
                  "generation": {"temperature": 0.0, "do_sample": False,
                                 "max_new_tokens": 1024}},
        "prompts": {"digest": "digest", "forensic_sha256": "prompt-hash"},
        "experts": {"probe_dino_expert": {
            "class": "ProbeDinoExpert", "source_name": "probe_dino_expert",
            "weights_sha256": "weights", "weights_bytes": 1}},
        "budget": {"halting_policy": "v3", "max_expert_calls": 5,
                   "max_model_turns": 6, "evidence_injection": "text+image"},
        "code": {"commit": "0" * 40, "dirty": False, "dirty_paths": []},
    }


def _prepare(tmp_path, *, metadata=None, decisions=None):
    batch = Batch(tmp_path, variant_ids=("v0", "v1"))
    for variant_id in batch.variant_ids:
        batch.add(variant_id)
    batch.finish()
    with open(batch.metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata or _metadata(batch), handle, ensure_ascii=False)
    ledger = _ledger(tmp_path, decisions if decisions is not None else _decisions(batch))
    return batch, ledger


def _register(batch, ledger, tmp_path, **overrides):
    options = dict(
        records_dir=str(batch.records_dir),
        ledger_path=str(ledger),
        manifest_path=str(batch.manifest_path),
        metadata_path=str(batch.metadata_path),
        accepted_path=str(tmp_path / "review" / "accepted_protocol.json"))
    options.update(overrides)
    return accept.register(GENERATION, BATCH, "hj", **options)


class TestRegistration:
    def test_it_writes_the_latest_identity_and_history(self, tmp_path, monkeypatch):
        monkeypatch.setattr(accept, "_verify_protocol_files_unchanged_since",
                            lambda *a, **k: [])
        batch, ledger = _prepare(tmp_path)
        payload = _register(batch, ledger, tmp_path)

        target = tmp_path / "review" / "accepted_protocol.json"
        history = tmp_path / "review" / accept.HISTORY_NAME
        assert target.exists() and history.exists()
        stored = json.loads(target.read_text(encoding="utf-8"))
        assert stored["identity_sha256"] == payload["identity_sha256"]
        assert stored["accepted_by"] == "hj"
        assert stored["coverage"] == {"records": 2, "expected": 2, "passed": 2,
                                      "extra_decisions": []}
        assert stored["identity"]["model"]["revision"] == "rev"
        assert stored["identity"]["evidence_pack"]["version"].startswith("slim/")
        assert stored["identity"]["experts"]["probe_dino_expert"][
            "preprocess_sha256"]

    def test_a_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(accept, "_verify_protocol_files_unchanged_since",
                            lambda *a, **k: [])
        batch, ledger = _prepare(tmp_path)
        payload = _register(batch, ledger, tmp_path, dry_run=True)
        assert payload["identity"]["model"]["revision"] == "rev"
        assert not (tmp_path / "review").exists()

    def test_incomplete_review_is_refused_before_writing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(accept, "_verify_protocol_files_unchanged_since",
                            lambda *a, **k: [])
        batch, _full = _prepare(tmp_path)
        decisions = _decisions(batch)
        partial = str(tmp_path / "partial.jsonl")
        from utils.review_store import append_ledger

        append_ledger(GENERATION, BATCH, decisions[:-1], path=partial)
        with pytest.raises(SystemExit) as caught:
            _register(batch, partial, tmp_path)
        assert "缺审" in str(caught.value)
        assert not (tmp_path / "review").exists()

    def test_a_protocol_file_changed_since_b2_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(accept, "_verify_protocol_files_unchanged_since",
                            lambda *a, **k: ["state_machine/controller.py: 自 b2 后已变"])
        batch, ledger = _prepare(tmp_path)
        with pytest.raises(SystemExit) as caught:
            _register(batch, ledger, tmp_path)
        assert "自 b2 后已变" in str(caught.value)
        assert not (tmp_path / "review").exists()

    def test_a_dirty_non_output_code_path_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(accept, "_verify_protocol_files_unchanged_since",
                            lambda *a, **k: [])
        metadata = _metadata(None)
        metadata["code"] = {"commit": "0" * 40, "dirty": True,
                            "dirty_paths": ["state_machine/controller.py"]}
        batch, ledger = _prepare(tmp_path, metadata=metadata)
        with pytest.raises(SystemExit) as caught:
            _register(batch, ledger, tmp_path)
        assert "dirty" in str(caught.value)

    def test_output_only_dirty_paths_are_a_warning_not_a_refusal(self, tmp_path, monkeypatch):
        monkeypatch.setattr(accept, "_verify_protocol_files_unchanged_since",
                            lambda *a, **k: [])
        metadata = _metadata(None)
        metadata["code"] = {"commit": "0" * 40, "dirty": True,
                            "dirty_paths": ["sft_data/g8/trajectories/b2/",
                                            "traces/sft_sessions/g8/b2/"]}
        batch, ledger = _prepare(tmp_path, metadata=metadata)
        payload = _register(batch, ledger, tmp_path)
        assert payload["warnings"]["b2_code_dirty"] is True
        assert payload["warnings"]["b2_dirty_paths_non_output"] == []

    def test_re_registration_requires_a_reason_and_keeps_history(self, tmp_path, monkeypatch):
        monkeypatch.setattr(accept, "_verify_protocol_files_unchanged_since",
                            lambda *a, **k: [])
        batch, ledger = _prepare(tmp_path)
        _register(batch, ledger, tmp_path)
        with pytest.raises(SystemExit) as caught:
            _register(batch, ledger, tmp_path)
        assert "--reason" in str(caught.value)
        _register(batch, ledger, tmp_path, reason="重新登记")
        lines = (tmp_path / "review" / accept.HISTORY_NAME).read_text(
            encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_missing_weight_identity_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(accept, "_verify_protocol_files_unchanged_since",
                            lambda *a, **k: [])
        metadata = _metadata(None)
        metadata["experts"]["probe_dino_expert"]["weights_sha256"] = ""
        batch, ledger = _prepare(tmp_path, metadata=metadata)
        with pytest.raises(SystemExit) as caught:
            _register(batch, ledger, tmp_path)
        assert "weights_sha256" in str(caught.value)

    def test_metadata_from_another_batch_is_refused(self, tmp_path, monkeypatch):
        monkeypatch.setattr(accept, "_verify_protocol_files_unchanged_since",
                            lambda *a, **k: [])
        metadata = _metadata(None)
        metadata["batch"] = "b1"
        batch, ledger = _prepare(tmp_path, metadata=metadata)
        with pytest.raises(SystemExit) as caught:
            _register(batch, ledger, tmp_path)
        assert "不是 g8/b2" in str(caught.value)


class TestTheGitProof:
    def test_unchanged_files_produce_no_problems(self, tmp_path, monkeypatch):
        from config import PROJECT_ROOT

        real = open(os.path.join(PROJECT_ROOT, "config.py"), "rb").read()
        monkeypatch.setattr(accept, "_commit_bytes",
                            lambda commit, relative: (real, ""))
        assert accept._verify_protocol_files_unchanged_since("0" * 40,
                                                              ["config.py"]) == []

    def test_a_changed_file_is_named(self, tmp_path, monkeypatch):
        monkeypatch.setattr(accept, "_commit_bytes",
                            lambda commit, relative: (b"old bytes", ""))
        problems = accept._verify_protocol_files_unchanged_since(
            "0" * 40, ["config.py"])
        assert problems and "config.py" in problems[0] and "已变" in problems[0]

    def test_a_file_missing_from_the_commit_is_refused(self, monkeypatch):
        monkeypatch.setattr(accept, "_commit_bytes",
                            lambda commit, relative: (None, "not found"))
        problems = accept._verify_protocol_files_unchanged_since(
            "0" * 40, ["config.py"])
        assert problems and "读不到" in problems[0]
