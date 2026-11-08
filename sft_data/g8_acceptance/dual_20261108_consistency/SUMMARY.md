# dual_20261108_consistency — automatic summary

> Descriptive only; not a human acceptance verdict. 24 sessions, same manifest.
> Qwen3-VL-8B-Instruct, greedy, `probe+trufor`; joint prompt digest
> `6b4756ca21bc7a63`; wall ~762 s (17:30:49–17:43:31).

## Headline

| 指标 | `dual_20261108_certainty` | `dual_20261108_peakcrop` | 本批 |
| --- | ---: | ---: | ---: |
| 总体判断正确 | 12 | 17 | **20** |
| 整图判断正确 | 14 | 16 | 14 |
| 局部判断正确 | 8 | 11 | 7 |
| 两维都正确 | 5 | 8 | 5 |
| Real/Not-tampered 被误判 Tampered | 4 | 2 | **1** |
| `joint_verdict_issues` 会话 | 7 | 0 | **0** |
| 覆盖检查失败/未纠正 | 1 | 0 | **0** |
| 覆盖检查后重答 | 4 | 4 | 3 |
| certainty 字段检查重答/失败 | 未记录 | 未记录 | 1 / **0** |
| 总体 Uncertain | 7 | 3 | **1** |
| certainty High/Medium/Low/null | 14/3/0/7 | 未记录 | 21/2/0/1 |
| expert call 数为 2 | 15 | 21 | 12 |
| 有执行失败专家 | 5 | 1 | 2 |
| 强跨任务相互验证措辞 | 1 | 1 | **0** |
| 全图 bbox | 0 | 0 | 0 |

- `certainty_v1` 24/24，`confidence` 字段 0 条。
- TruFor 失败 2 条：`Midjourney`、`42005_2`；其余测量有效。
- 总体错误 4 条：`339e8fae`、`51136`、`53059`（真实图被 PROBE 高分带成 Fake）、
  `57501_0`（真实局部篡改但整体 Uncertain）。前三条的局部维度均保持 Uncertain，
  没有再把真实图写成局部 Tampered。
- 局部漏检（真实 Tampered 写成 Uncertain/Not tampered）：`39890_1`、`72399_2`、
  `42005_0`、`42005_2`、`57501_0`；其中 `42005_2` 是 TruFor 失败。
- 局部误报仅 1 条：`ADM`（真值 local Not applicable，模型写 Tampered）；`51136`、
  `53059`、`339e8fae` 已从 Tampered 降为 Uncertain。

## What changed

- 新增一次 certainty 字段纠错：`certainty_check` 保留原文/反馈/重答；Uncertain 必须
  null，不静默转换。
- 总体一致性可纠正：Uncertain 与确定 Fake 分项不能并存；总体 Real 不能仅凭局部
  “未发现篡改”且 PROBE 可用未测；`Not applicable` 只限整图 AI-generated。
- 局部候选/低分语义：TruFor 仅候选/非证明时不得直接写 Tampered；有效低分且无高
  响应区域支持 Not tampered，不能只因“不能证明不存在”写 Uncertain。
- 提示词明确 PROBE/TruFor 逻辑独立、未校准不等于无信息、High 不得与主要限制冲突，
  并禁止 reasoning/report 写“交叉验证/相互验证/共同支持/相互支持”。
- `final_v2` 对 failed/not_corrected 的 coverage/certainty 检查拒收，并将审计块
  保留到 rendered sample，避免校验器看不到门。

## Remaining

- certainty 仍偏高：21/23 个确定结论为 High、0 个 Low；High 是否机械仍需人工判断。
- 局部召回偏低（7/24）：模型对高于阈值的 TruFor 候选倾向 Uncertain，导致真实
  Tampered 漏检；这是保守取舍，不是 schema 问题。
- 4 条总体错误中 3 条来自 PROBE 对真实图的高分误报；需要人工核对 `339e8fae`、
  `51136`、`53059` 的 PROBE/图像关系。
- `ADM` 的 local Tampered 与真值 Not applicable 不一致；`39890_1` 的局部高分没有
  转成 Tampered，均需人工判断。
- 2 条 TruFor 外部失败（`Midjourney`、`42005_2`）不能归因于提示词。

## Decision

Pending human review; **not SFT, not expanding to new sources**. The automatic
counts are descriptive only; the HTML workbench and original traces are
authoritative.
