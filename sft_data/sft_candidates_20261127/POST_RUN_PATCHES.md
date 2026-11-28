# 首轮 SFT 候选样板 — 运行后文字补丁（第一轮历史）

日期：2026-11-28。未加载 GPU、未重跑专家、未改任何工具输出。
**说明**：本文件记录第一轮 24 条交付时的 Midjourney 文字补丁；2026-11-28 的判例修订
已重新生成 26 条轨迹，`candidate_scripts.json` 中该条已直接使用修正后的定性表述，
本文件仅保留为第一轮审计历史。

## 补丁 1：`genimage__Midjourney__229_midjourney_197__q95`

- **原句**（候选推理）：
  “…可靠性图在狗身区域明显偏低（蓝色/青色），低可靠占比约 **0.39**；该局部响应不能作为独立篡改证据。”
- **改为**：
  “…可靠性图在狗身区域明显偏低（蓝色/青色）；该局部响应不能作为独立篡改证据。”
- **原因**：TruFor 投递给模型的 slim token 只有 score、reading 和图像 artifact，没有
  数字 `map_stats`；`0.39` 来自预筛 CSV，助手当时并未收到这个数字。它违反“每一步只引用
  当时已经收到的证据”，因此删除数字，保留可靠性图的定性读数。
- **范围**：只改助手文字。工具输出、evidence_id、score、bbox、调用顺序、总体/整图/局部
  结论和 certainty 均未变。
- **落盘**：`traces/forensic_sft_session_20261127_153040_229_midjourney_197.json` 的
  `conversations[7].value` 与 `metadata.model_calls[2].output`；页面已重新渲染；
  `candidate_scripts.json` 已用修正后的源文本重新生成；`records.json` 的该条记录增加
  `post_run_text_patch`；`batch_meta.json` 记录旧/新 scripts sha256 和补丁范围。

## 审计结果

`AUDIT_REPORT.md` 在补丁后 24/24 pass、0 问题。审计覆盖：证据 ID 是否在引用前已投递、
数字是否来自已投递 token、助手文本是否出现真值/mask 词、结论字段是否与 record 一致、
页面是否含原图/三图/分数表/调用顺序/结案/可用链接，以及 8/6/6/4 组成和训练侧划分。
