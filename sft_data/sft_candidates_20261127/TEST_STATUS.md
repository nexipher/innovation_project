# 首轮 SFT 候选样板 — 收尾测试状态

日期：2026-11-28。未开启 GPU；以下均为 CPU 测试/只读审计。

## 本轮相关

| 测试 | 结果 |
|---|---|
| `pytest -q tests/test_scripted_candidate.py tests/test_g8_review_workstation.py tests/test_controller.py tests/test_controller_v3.py` | **107 passed** |
| `bash scripts/run_tests.sh core` | **834 passed, 1 skipped**（50.91 s） |
| `bash scripts/run_tests.sh module` | **1434 passed, 5 failed, 1 skipped**（129.69 s） |

## module 的 5 个失败

这 5 个失败在候选交付提交 `75b13ad` 的独立 worktree 中**逐条复现**，不是本轮
`mllm/scripted_candidate.py`、候选页面或收尾审计引入的：

- `tests/test_evidence_check_session.py::...test_a_fabrication_inside_an_accepted_candidate_is_caught`
- `tests/test_evidence_check_session.py::...test_the_counters_separate_the_four_states`
  - 期望 `pass_after_correction`，当前证据检查返回 `pass`；是既有测试期望与当前
    certainty/证据检查行为不一致。
- `tests/test_expert_admission.py::...test_ela_is_not_registered`
- `tests/test_expert_admission.py::...test_call_tags_map_to_shipped_sources`
  - 测试仍期望 `EXPERT_KEY_MAP` 不含 `trufor`；TruFor 映射是既有双专家接入。
- `tests/test_model_call_log.py::...test_a_client_with_the_api_lands_in_the_trace`
  - `QwenVLClient` 的 fake inference 收到空 messages；是既有客户端测试的脆弱假设。

## 边界

- 本轮没有修改这 5 个测试或它们对应的运行行为；建议单独做一次“退役/更新既有测试”
  的维护提交，而不是在候选样板收尾里顺手改掉。
- 没有 GPU 依赖的测试被跳过或未运行；没有把 Mock 结果冒充真实专家。
