"""
外部 Agent 的启动门禁（2026-10-18 用户指定，plan §4.12b）。

门禁不数决定条数：以记录目录里的真实 `trajectory_id` 为准，逐个与台账最新决定
匹配；指纹用工作台同一个 `sample_fingerprint(record, trace)` 现场重算；台账的
`stale` 字段不是放行依据。协议身份的比较在 `tests/test_protocol_identity.py`。

所有夹具都在 tmp_path 下——测试不往仓库的批次目录写任何东西。
"""

import json
import os

import pytest

from utils.acceptance import verify_review_coverage
from utils.batch import assert_protocol_accepted
from utils.protocol_identity import SCHEMA, canonical_identity, identity_sha256
from utils.review_store import Decision, append_ledger, sample_fingerprint

pytestmark = pytest.mark.core

POLICY = "g8_default"
GENERATION = "g8"
BATCH = "b2"


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------

class Batch:
    def __init__(self, tmp_path, variant_ids=("v0", "v1")):
        self.root = tmp_path
        self.records_dir = tmp_path / "trajectories"
        self.records_dir.mkdir()
        self.trace_dir = tmp_path / "traces"
        self.trace_dir.mkdir()
        self.variant_ids = list(variant_ids)
        self.manifest_path = tmp_path / "variants.json"
        self.metadata_path = self.records_dir / "metadata.json"
        self.records = {}
        self.traces = {}
        self.fingerprints = {}

    def add(self, variant_id, *, duplicate_filename=None, write_trace=True,
            trace_override=None, record_extra=None):
        trajectory_id = f"{POLICY}__{variant_id}"
        trace_path = trace_override
        if trace_path is None:
            trace_path = str(self.trace_dir / f"{trajectory_id}.json")
        record = {
            "trajectory_id": trajectory_id, "variant_id": variant_id,
            "source_id": f"src/{variant_id}", "treatment": "png",
            "container": "png", "generator": "g", "split": "train",
            "ground_truth": "Fake", "final_verdict": "Fake",
            "verdict_status": "model_concluded", "confidence": 0.9,
            "halting_reason": "model_concluded", "policy_reasons": [],
            "report": "r", "counters": {"model_turns": 1},
            "evidence_check": {}, "tools_served": ["probe_dino_expert"],
            "record_schema": "g8_r1", "image_path": f"images/{variant_id}.png",
            "trace_path": trace_path,
        }
        record.update(record_extra or {})
        trace = {"conversations": [
            {"from": "human", "value": f"image prompts for {variant_id}"},
            {"from": "gpt", "value": f"answer {variant_id}"}],
            "metadata": {"model_turn_count": 1, "expert_call_count": 1}}
        if write_trace:
            with open(trace_path, "w", encoding="utf-8") as handle:
                json.dump(trace, handle, ensure_ascii=False)
        name = duplicate_filename or f"{trajectory_id}.json"
        with open(self.records_dir / name, "w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False)
        self.records[trajectory_id] = record
        self.traces[trajectory_id] = trace
        self.fingerprints[trajectory_id] = sample_fingerprint(record, trace)
        return trajectory_id

    def finish(self, *, manifest_variant_ids=None):
        variants = {}
        for variant_id in (manifest_variant_ids or self.variant_ids):
            variants[f"sft_data/{variant_id}.png"] = {"variant_id": variant_id}
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump({"variants": variants}, handle, ensure_ascii=False)
        with open(self.metadata_path, "w", encoding="utf-8") as handle:
            json.dump({"generation": GENERATION, "batch": BATCH,
                       "policy": POLICY, "variants_total": len(variants)},
                      handle, ensure_ascii=False)
        return self


def _frozen(tmp_path, *, registered=True, raw=None):
    path = tmp_path / "FROZEN.json"
    if raw is not None:
        path.write_text(raw, encoding="utf-8")
    else:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"frozen": ([{"generation": GENERATION, "batch": BATCH}]
                                  if registered else [])}, handle)
    return str(path)


def _ledger(tmp_path, decisions, *, generation=GENERATION, batch=BATCH):
    path = str(tmp_path / "dispositions.jsonl")
    append_ledger(generation, batch, decisions, path=path)
    return path


def _decisions(batch, decision="pass", **overrides):
    return [Decision(trajectory_id=trajectory_id,
                     fingerprint=overrides.get("fingerprint",
                                               batch.fingerprints[trajectory_id]),
                     decision=overrides.get("decision", decision),
                     reviewer=overrides.get("reviewer", "hj"),
                     note=overrides.get("note", ""))
            for trajectory_id in batch.records]


def _identity():
    return canonical_identity(
        model={"checkpoint": "m", "revision": "r", "dtype": "bfloat16",
               "inference_mode": "non-thinking",
               "generation": {"temperature": 0.0, "do_sample": False,
                              "max_new_tokens": 1024}},
        prompts={"digest": "d", "forensic_sha256": "f"},
        experts={"probe_dino_expert": {
            "class": "ProbeDinoExpert", "source_name": "probe_dino_expert",
            "weights_sha256": "w", "weights_bytes": 1,
            "preprocess": {"sha256": "p", "detail": "x"}}},
        tools={"policy": POLICY, "policy_version": "v3",
               "tools_served": ["probe_dino_expert"],
               "evidence_injection": "text+image"},
        budget={"halting_policy": "v3", "max_expert_calls": 5,
                "max_model_turns": 6, "evidence_injection": "text+image"},
        decoding={"greedy": True, "inference_mode": "non-thinking"},
        provenance={"files": {"a.py": "h"}, "sha256": "x"})


def _accepted(tmp_path, identity=None):
    identity = identity or _identity()
    path = tmp_path / "accepted_protocol.json"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"schema": SCHEMA, "generation": GENERATION, "batch": BATCH,
                   "identity": identity,
                   "identity_sha256": identity_sha256(identity)},
                  handle, ensure_ascii=False)
    return str(path)


def _gate(batch, tmp_path, *, frozen=None, ledger=None, accepted=None):
    return assert_protocol_accepted(
        GENERATION, BATCH, "外部评测",
        records_dir=str(batch.records_dir),
        frozen_path=frozen or _frozen(tmp_path),
        ledger_path=ledger or _ledger(tmp_path, _decisions(batch)),
        accepted_path=accepted or _accepted(tmp_path),
        manifest_path=str(batch.manifest_path),
        metadata_path=str(batch.metadata_path))


# ---------------------------------------------------------------------------
# 通过路径
# ---------------------------------------------------------------------------

class TestItPassesOnlyWhenEverythingMatches:
    def test_a_complete_matching_batch_passes_and_returns_the_identity(self, tmp_path):
        batch = Batch(tmp_path)
        for variant_id in batch.variant_ids:
            batch.add(variant_id)
        batch.finish()
        payload = _gate(batch, tmp_path)
        assert payload["schema"] == SCHEMA
        assert payload["identity"]["model"]["checkpoint"] == "m"

    def test_the_ledger_stale_flag_is_ignored_and_the_current_fingerprint_decides(self, tmp_path):
        """台账说 stale=True，但现场指纹与决定一致 → 决定针对的就是当前内容。"""
        batch = Batch(tmp_path)
        for variant_id in batch.variant_ids:
            batch.add(variant_id)
        batch.finish()
        decisions = [
            Decision(trajectory_id=trajectory_id,
                     fingerprint=batch.fingerprints[trajectory_id],
                     decision="pass", reviewer="hj", stale=True,
                     stale_reason="重建页面时标过")
            for trajectory_id in batch.records]
        _gate(batch, tmp_path, ledger=_ledger(tmp_path, decisions))

    def test_a_decision_with_a_stale_flag_false_but_wrong_fingerprint_is_refused(self, tmp_path):
        batch = Batch(tmp_path)
        for variant_id in batch.variant_ids:
            batch.add(variant_id)
        batch.finish()
        decisions = [Decision(trajectory_id=trajectory_id, fingerprint="old",
                              decision="pass", reviewer="hj", stale=False)
                     for trajectory_id in batch.records]
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=_ledger(tmp_path, decisions))
        message = str(caught.value)
        assert "审核指纹与当前内容不一致" in message
        assert "台账 old，当前" in message


# ---------------------------------------------------------------------------
# 逐样本覆盖：ID 必须逐项对上
# ---------------------------------------------------------------------------

class TestPerSampleCoverage:
    def test_a_missing_decision_is_named_by_id(self, tmp_path):
        batch = Batch(tmp_path)
        for variant_id in batch.variant_ids:
            batch.add(variant_id)
        batch.finish()
        first = next(iter(batch.records))
        decisions = [Decision(trajectory_id=first,
                              fingerprint=batch.fingerprints[first],
                              decision="pass", reviewer="hj")]
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=_ledger(tmp_path, decisions))
        missing = [tid for tid in batch.records if tid != first][0]
        assert "缺审" in str(caught.value) and missing in str(caught.value)

    def test_another_samples_decision_cannot_fill_the_gap(self, tmp_path):
        """数量相同也可能缺审：台账多一条别的 ID，不能抵扣。"""
        batch = Batch(tmp_path)
        for variant_id in batch.variant_ids:
            batch.add(variant_id)
        batch.finish()
        first = next(iter(batch.records))
        decisions = [Decision(trajectory_id=first,
                              fingerprint=batch.fingerprints[first],
                              decision="pass", reviewer="hj"),
                     Decision(trajectory_id="g8_default__other", fingerprint="x",
                              decision="pass", reviewer="hj")]
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=_ledger(tmp_path, decisions))
        message = str(caught.value)
        assert "缺审" in message
        assert "g8_default__other" in message           # 单独列出，不算覆盖
        assert "不计入覆盖" in message

    def test_extra_decisions_are_reported_but_do_not_fill_or_block(self, tmp_path):
        """历史遗留的别的 ID：不抵扣覆盖，也不单独阻断（台账是追加式的）。"""
        batch = Batch(tmp_path)
        for variant_id in batch.variant_ids:
            batch.add(variant_id)
        batch.finish()
        decisions = _decisions(batch)
        decisions.append(Decision(trajectory_id="g8_default__retired",
                                  fingerprint="x", decision="pass", reviewer="hj"))
        result = verify_review_coverage(
            GENERATION, BATCH, records_dir=str(batch.records_dir),
            ledger_path=_ledger(tmp_path, decisions),
            manifest_path=str(batch.manifest_path),
            metadata_path=str(batch.metadata_path))
        assert result["ok"] is True
        assert result["extra_decisions"] == ["g8_default__retired"]

    def test_duplicate_record_ids_are_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.add("v0", duplicate_filename="second.json")
        batch.finish()
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path)
        assert "重复记录 ID" in str(caught.value)

    def test_an_empty_batch_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.finish()
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path)
        assert "批次为空" in str(caught.value)

    def test_a_malformed_record_is_refused_by_filename(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        (batch.records_dir / "broken.json").write_text("{not json", encoding="utf-8")
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path)
        assert "broken.json" in str(caught.value) and "无法解析" in str(caught.value)

    def test_a_record_without_trajectory_id_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        (batch.records_dir / "summary.json").write_text(
            json.dumps({"report": "not a session"}), encoding="utf-8")
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path)
        assert "summary.json" in str(caught.value) and "缺 trajectory_id" in str(caught.value)

    def test_metadata_from_another_batch_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        metadata = json.loads(batch.metadata_path.read_text(encoding="utf-8"))
        metadata["batch"] = "b1"
        batch.metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path)
        assert "不是 g8/b2" in str(caught.value)

    def test_a_deleted_record_is_caught_by_the_expected_variant_set(self, tmp_path):
        """记录被删一条时，逐 ID 覆盖会以为'当前样本变少了'；期望集必须兜底。"""
        batch = Batch(tmp_path, variant_ids=("v0", "v1"))
        batch.add("v0")                       # v1 的记录被删掉了
        batch.finish(manifest_variant_ids=("v0", "v1"))
        decisions = _decisions(batch)         # 现有记录全部通过
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=_ledger(tmp_path, decisions))
        assert "记录缺失" in str(caught.value) and f"{POLICY}__v1" in str(caught.value)

    def test_an_unexpected_record_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.add("v1")                       # 清单里没有 v1
        batch.finish(manifest_variant_ids=("v0",))
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=_ledger(tmp_path, _decisions(batch)))
        assert "记录多出" in str(caught.value) and f"{POLICY}__v1" in str(caught.value)


# ---------------------------------------------------------------------------
# trace 与指纹
# ---------------------------------------------------------------------------

class TestTraceAndFingerprint:
    def test_a_missing_trace_is_refused_by_id(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0", trace_override=str(tmp_path / "nope.json"),
                  write_trace=False)
        batch.finish()
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path)
        assert f"{POLICY}__v0" in str(caught.value)
        assert "trace 文件不存在" in str(caught.value)

    def test_a_malformed_trace_is_refused_by_id(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        (batch.trace_dir / f"{POLICY}__v0.json").write_text("{oops", encoding="utf-8")
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path)
        assert f"{POLICY}__v0" in str(caught.value)
        assert "trace 无法解析" in str(caught.value)

    def test_changing_the_trace_changes_the_fingerprint(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        trajectory_id = batch.add("v0")
        batch.finish()
        decisions = _decisions(batch)
        # 审核之后 trace 内容变了（增加一轮）→ 旧决定必须失效
        trace = batch.traces[trajectory_id]
        trace["conversations"].append({"from": "gpt", "value": "extra turn"})
        (batch.trace_dir / f"{trajectory_id}.json").write_text(
            json.dumps(trace), encoding="utf-8")
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=_ledger(tmp_path, decisions))
        assert "审核指纹与当前内容不一致" in str(caught.value)


# ---------------------------------------------------------------------------
# 台账本身
# ---------------------------------------------------------------------------

class TestTheLedgerItself:
    def test_a_missing_ledger_is_refused_explicitly(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=str(tmp_path / "missing.jsonl"))
        assert "验收台账不存在" in str(caught.value)

    def test_an_empty_ledger_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        path = tmp_path / "empty.jsonl"
        path.write_text("", encoding="utf-8")
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=str(path))
        assert "验收台账为空" in str(caught.value)

    def test_a_malformed_ledger_line_is_refused_with_its_line_number(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        path = tmp_path / "ledger.jsonl"
        path.write_text("{broken\n", encoding="utf-8")
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=str(path))
        assert "第 1 行无法解析" in str(caught.value)

    def test_a_ledger_line_from_another_batch_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        path = str(tmp_path / "cross.jsonl")
        append_ledger("g7", "b1", _decisions(batch), path=path)
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=path)
        assert "属于 g7/b1" in str(caught.value)

    def test_revise_and_reject_are_listed_separately(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0", "v1"))
        for variant_id in batch.variant_ids:
            batch.add(variant_id)
        batch.finish()
        ordered = list(batch.records)
        decisions = [
            Decision(trajectory_id=ordered[0],
                     fingerprint=batch.fingerprints[ordered[0]],
                     decision="revise", reviewer="hj", note="措辞"),
            Decision(trajectory_id=ordered[1],
                     fingerprint=batch.fingerprints[ordered[1]],
                     decision="reject", reviewer="hj", note="不忠实"),
        ]
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=_ledger(tmp_path, decisions))
        message = str(caught.value)
        assert "未通过（revise）" in message and ordered[0] in message
        assert "未通过（reject）" in message and ordered[1] in message

    def test_an_invalid_decision_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        trajectory_id = batch.add("v0")
        batch.finish()
        decisions = [Decision(trajectory_id=trajectory_id,
                              fingerprint=batch.fingerprints[trajectory_id],
                              decision="pass", reviewer="")]   # 无审核人
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, ledger=_ledger(tmp_path, decisions))
        assert "决定字段不合法" in str(caught.value) and trajectory_id in str(caught.value)


# ---------------------------------------------------------------------------
# 冻结登记表与验收身份文件
# ---------------------------------------------------------------------------

class TestFreezeAndIdentityFile:
    def test_a_missing_frozen_file_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, frozen=str(tmp_path / "nope.json"))
        assert "冻结登记表不存在" in str(caught.value)

    def test_a_broken_frozen_file_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        path = tmp_path / "FROZEN.json"
        path.write_text("{broken", encoding="utf-8")
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, frozen=str(path))
        assert "冻结登记表无法读取" in str(caught.value)

    def test_an_unfrozen_batch_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, frozen=_frozen(tmp_path, registered=False))
        assert "还没有冻结" in str(caught.value)

    def test_a_missing_accepted_identity_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, accepted=str(tmp_path / "missing.json"))
        assert "尚未登记" in str(caught.value)

    def test_a_wrong_identity_schema_is_refused(self, tmp_path):
        batch = Batch(tmp_path, variant_ids=("v0",))
        batch.add("v0")
        batch.finish()
        path = tmp_path / "accepted.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"schema": "other/1", "identity": {}}, handle)
        with pytest.raises(SystemExit) as caught:
            _gate(batch, tmp_path, accepted=str(path))
        assert "schema" in str(caught.value)


# ---------------------------------------------------------------------------
# 批次入口：门禁失败不留产物
# ---------------------------------------------------------------------------

class TestTheBatchEntryOrdering:
    def _manifest(self, tmp_path):
        with open(tmp_path / "inputs.json", "w", encoding="utf-8") as handle:
            json.dump({"variants": {
                "img.png": {"variant_id": "v", "source_id": "s", "label": "Fake",
                            "generator": "g", "split": "train",
                            "resolution": [8, 8], "treatment": "png",
                            "container": "png", "quality": None}}},
                handle)
        return str(tmp_path / "inputs.json")

    def test_a_first_phase_failure_leaves_no_batch_directory(self, tmp_path, monkeypatch):
        from scripts import g8_generate_batch as entry

        monkeypatch.setattr(entry, "assert_protocol_accepted",
                            lambda *a, **k: (_ for _ in ()).throw(
                                SystemExit("门禁")))
        out = tmp_path / "out"
        with pytest.raises(SystemExit) as caught:
            entry.run_batch("g8", "ext_probe", self._manifest(tmp_path),
                            "g8_default", 1, True, False, True,
                            "non-thinking", False, out_dir=str(out),
                            purpose="external_evaluation_only")
        assert "门禁" in str(caught.value)
        assert not out.exists()

    def test_a_second_phase_mismatch_leaves_no_batch_directory(self, tmp_path, monkeypatch):
        """协议身份在管线构建后比较；不一致时目录还没创建。"""
        from scripts import g8_generate_batch as entry
        from utils import protocol_identity

        accepted = _identity()
        monkeypatch.setattr(entry, "assert_protocol_accepted",
                            lambda *a, **k: {"schema": SCHEMA,
                                             "identity": accepted,
                                             "identity_sha256": identity_sha256(accepted)})
        import main as entry_module

        monkeypatch.setattr(entry_module, "build_pipeline",
                            lambda *a, **k: type("Base", (), {
                                "_mllm": None, "_experts": {} })())
        changed = json.loads(json.dumps(accepted))
        changed["model"]["revision"] = "different"
        monkeypatch.setattr(protocol_identity, "runtime_identity",
                            lambda *a, **k: changed)
        out = tmp_path / "out2"
        with pytest.raises(SystemExit) as caught:
            entry.run_batch("g8", "ext_probe", self._manifest(tmp_path),
                            "g8_default", 1, False, False, True,
                            "non-thinking", False, out_dir=str(out),
                            purpose="external_evaluation_only")
        assert "model.revision" in str(caught.value)
        assert not out.exists()

    def test_other_purposes_do_not_hit_the_gate(self, tmp_path, monkeypatch):
        from scripts import g8_generate_batch as entry

        called = {}
        monkeypatch.setattr(entry, "assert_protocol_accepted",
                            lambda *a, **k: called.setdefault("hit", True))
        # build_pipeline 也要挡住：这条测试只关心门禁不被调用。
        import main as entry_module

        monkeypatch.setattr(entry_module, "build_pipeline",
                            lambda *a, **k: (_ for _ in ()).throw(
                                SystemExit("不应加载模型")))
        with pytest.raises(SystemExit) as caught:
            entry.run_batch("g8", "b_dry", self._manifest(tmp_path), "g8_default",
                            1, True, False, True, "non-thinking", False,
                            out_dir=str(tmp_path / "out3"), purpose="dry_run")
        assert "不应加载模型" in str(caught.value)
        assert "hit" not in called
