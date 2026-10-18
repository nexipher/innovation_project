"""
The hard pause before the external Agent run (user, 2026-10-18).

Order: b2 runs → human acceptance → the user freezes the protocol → only then the
external Agent's 200 sessions may run.  "Run the whole chain once the GPU is on"
would silently skip the middle two steps, so the precondition is a gate, not a
sentence in a plan: the batch entry refuses unless b2 is **both** registered as
frozen and fully passed in its acceptance ledger.

All fixtures live under tmp_path - nothing here touches the repository.
"""

import json
import os

import pytest

from utils.batch import assert_protocol_accepted
from utils.review_store import Decision, append_ledger, sample_fingerprint

pytestmark = pytest.mark.core

RECORD = {"trajectory_id": "a", "variant_id": "a", "ground_truth": "Fake",
          "final_verdict": "Fake", "treatment": "png"}
TRACE = {"conversations": [{"from": "gpt", "value": "x"}]}
FINGERPRINT = sample_fingerprint(RECORD, TRACE)


def _frozen(tmp_path, registered: bool):
    path = tmp_path / "FROZEN.json"
    payload = {"frozen": ([{"generation": "g8", "batch": "b2", "reason": "验收通过"}]
                          if registered else [])}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def _records(tmp_path, count: int = 2):
    directory = tmp_path / "trajectories"
    directory.mkdir(exist_ok=True)
    for index in range(count):
        path = directory / f"r{index}.json"
        path.write_text(json.dumps({"trajectory_id": f"r{index}"}), encoding="utf-8")
    return str(directory)


def _ledger(tmp_path, decisions):
    path = str(tmp_path / "dispositions.jsonl")
    append_ledger("g8", "b2", decisions, path=path)
    return path


def _gate(tmp_path, *, frozen, decisions, records=2):
    assert_protocol_accepted("g8", "b2", "外部评测",
                             records_dir=_records(tmp_path, records),
                             frozen_path=_frozen(tmp_path, frozen),
                             ledger_path=_ledger(tmp_path, decisions))


class TestTheGate:
    def test_an_unfrozen_protocol_is_refused(self, tmp_path):
        with pytest.raises(SystemExit) as caught:
            _gate(tmp_path, frozen=False, decisions=[])
        assert "还没有冻结" in str(caught.value)

    def test_a_frozen_protocol_with_a_complete_pass_ledger_passes(self, tmp_path):
        decisions = [Decision(trajectory_id=f"r{i}", fingerprint=FINGERPRINT,
                              decision="pass", reviewer="hj") for i in range(2)]
        _gate(tmp_path, frozen=True, decisions=decisions)     # 不抛 = 通过

    def test_an_outstanding_revise_blocks_the_run(self, tmp_path):
        decisions = [Decision(trajectory_id="r0", fingerprint=FINGERPRINT,
                              decision="revise", reviewer="hj", note="措辞"),
                     Decision(trajectory_id="r1", fingerprint=FINGERPRINT,
                              decision="pass", reviewer="hj")]
        with pytest.raises(SystemExit) as caught:
            _gate(tmp_path, frozen=True, decisions=decisions)
        assert "验收尚未通过" in str(caught.value) and "r0" in str(caught.value)

    def test_a_stale_decision_blocks_the_run(self, tmp_path):
        """过期的"通过"不等于当前样本通过：内容变过就必须重审。"""
        decisions = [Decision(trajectory_id="r0", fingerprint="old",
                              decision="pass", reviewer="hj", stale=True,
                              stale_reason="内容已变"),
                     Decision(trajectory_id="r1", fingerprint=FINGERPRINT,
                              decision="pass", reviewer="hj")]
        with pytest.raises(SystemExit) as caught:
            _gate(tmp_path, frozen=True, decisions=decisions)
        assert "r0" in str(caught.value)

    def test_an_incomplete_ledger_blocks_the_run(self, tmp_path):
        decisions = [Decision(trajectory_id="r0", fingerprint=FINGERPRINT,
                              decision="pass", reviewer="hj")]
        with pytest.raises(SystemExit) as caught:
            _gate(tmp_path, frozen=True, decisions=decisions, records=2)
        assert "1/2" in str(caught.value)


class TestTheBatchEntryCallsTheGate:
    def _manifest(self, tmp_path):
        from tests.test_g8_batch_entry import IMAGE

        path = tmp_path / "manifest.json"
        path.write_text(json.dumps({"variants": {
            IMAGE: {"variant_id": "v", "source_id": "s", "label": "Fake",
                    "generator": "g", "split": "train", "resolution": [8, 8],
                    "treatment": "png", "container": "png", "quality": None}}}),
            encoding="utf-8")
        return str(path)

    def test_an_external_run_is_stopped_before_anything_loads(self, tmp_path, monkeypatch):
        from scripts import g8_generate_batch as entry

        called = {}

        def _stop(*args, **kwargs):
            called["hit"] = True
            raise SystemExit("门禁")

        monkeypatch.setattr(entry, "assert_protocol_accepted", _stop)
        with pytest.raises(SystemExit) as caught:
            entry.run_batch("g8", "ext_probe", self._manifest(tmp_path), "g8_default",
                            1, True, False, True, "non-thinking", False,
                            out_dir=str(tmp_path / "out"),
                            purpose="external_evaluation_only")
        assert called.get("hit") is True and "门禁" in str(caught.value)

    def test_other_purposes_do_not_hit_the_gate(self, tmp_path, monkeypatch):
        from scripts import g8_generate_batch as entry

        called = {}
        monkeypatch.setattr(entry, "assert_protocol_accepted",
                            lambda *a, **k: called.setdefault("hit", True))
        entry.run_batch("g8", "b_dry", self._manifest(tmp_path), "g8_default",
                        1, True, False, True, "non-thinking", False,
                        out_dir=str(tmp_path / "out2"), purpose="dry_run")
        assert "hit" not in called
