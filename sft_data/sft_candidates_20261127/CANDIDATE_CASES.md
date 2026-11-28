# 首轮 SFT 候选样板：重点案例与修订清单

完整 26 条（24 条主样板 + 2 条单专家样板）见 `index.html`。
助手文本是候选写手拟写，不是 Qwen 输出；PROBE/TruFor 与三图是真实运行。
真值和 mask 只用于审核，助手每一步只引用当时已收到的证据。

## 重点案例

| # | sample | 类型 | 真值 whole/local | 候选 whole/local/overall | PROBE | TruFor | 重点看什么 |
|---|---|---|---|---|---:|---:|---|
| 1 | `autosplice__54029__1__q95` | strong_local_evidence | Real / Tampered | Uncertain / Tampered / Fake | 0.721901 | 0.999517 | 强局部证据；PROBE 中高分但整图来源无独立支撑，整图 Uncertain、局部 Tampered。 |
| 2 | `autosplice__55310__1__q95` | strong_local_evidence | Real / Tampered | Real / Tampered / Fake | 0.078204 | 0.999163 | 对照：PROBE 低分支持整图 Real，局部 Tampered；保留 Real + Tampered。 |
| 3 | `autosplice__45749__1__q95` | strong_local_evidence | Real / Tampered | Uncertain / Tampered / Fake | 0.999415 | 0.998262 | 企鹅：PROBE 接近 1 但整图来源缺独立支撑；整图 Uncertain、局部 Tampered、总体 Fake。 |
| 4 | `autosplice__69868__0__q95` | strong_local_evidence | Real / Tampered | Uncertain / Tampered / Fake | 0.810086 | 0.995207 | 强局部证据、局部框较宽；整图 Uncertain + 局部 Tampered + 总体 Fake。 |
| 5 | `autosplice__48251__0__q95` | strong_local_evidence | Real / Tampered | Uncertain / Tampered / Fake | 0.930811 | 0.993394 | PROBE 高分的人像局部编辑；整图 Uncertain，局部框较宽。 |
| 6 | `autosplice__51186__authentic__q95` | trufor_high_real | Real / Not tampered | Real / Not tampered / Real | 0.000812 | 0.972335 | TruFor 0.97 但高响应只在画框右缘；局部 Not tampered，certainty Medium。 |
| 7 | `autosplice__65547__authentic__q95` | trufor_high_real | Real / Not tampered | Real / Uncertain / Uncertain | 0.037187 | 0.897234 | TruFor 0.90 的真图；高响应在肩部/衣领，内容解释不唯一，保留 Uncertain。 |
| 8 | `genimage__SD15__459_sdv5_00195__q95` | whole_generated | AI-generated / Not applicable | AI-generated / Not applicable / Fake | 0.999973 | 0.979874 | 整图生成 + TruFor 0.98 局部高响应；整图 AI-generated 后局部 Not applicable。 |

## 单专家样板

| # | sample | 调用 | 真值 overall | 候选 whole/local/overall | 审核页 |
|---|---|---|---|---|---|
| 9 | `single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | probe | Fake | AI-generated / Not applicable / Fake | [009__single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95.html](009__single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95.html) |
| 10 | `single_trufor__autosplice__54029__1__q95` | trufor | Fake | Uncertain / Tampered / Fake | [010__single_trufor__autosplice__54029__1__q95.html](010__single_trufor__autosplice__54029__1__q95.html) |

## 全部 26 条

| # | sample | 类型 | 真值 overall | 候选 overall | calls | 审核页 |
|---|---|---|---|---|---|---|
| 1 | `autosplice__54029__1__q95` | strong_local_evidence | Fake | Fake | trufor → probe | [001__autosplice__54029__1__q95.html](001__autosplice__54029__1__q95.html) |
| 2 | `autosplice__55310__1__q95` | strong_local_evidence | Fake | Fake | trufor → probe | [002__autosplice__55310__1__q95.html](002__autosplice__55310__1__q95.html) |
| 3 | `autosplice__45749__1__q95` | strong_local_evidence | Fake | Fake | probe → trufor | [003__autosplice__45749__1__q95.html](003__autosplice__45749__1__q95.html) |
| 4 | `autosplice__69868__0__q95` | strong_local_evidence | Fake | Fake | trufor → probe | [004__autosplice__69868__0__q95.html](004__autosplice__69868__0__q95.html) |
| 5 | `autosplice__48251__0__q95` | strong_local_evidence | Fake | Fake | probe → trufor | [005__autosplice__48251__0__q95.html](005__autosplice__48251__0__q95.html) |
| 6 | `autosplice__51186__authentic__q95` | trufor_high_real | Real | Real | trufor → probe | [006__autosplice__51186__authentic__q95.html](006__autosplice__51186__authentic__q95.html) |
| 7 | `autosplice__65547__authentic__q95` | trufor_high_real | Real | Uncertain | trufor → probe | [007__autosplice__65547__authentic__q95.html](007__autosplice__65547__authentic__q95.html) |
| 8 | `genimage__SD15__459_sdv5_00195__q95` | whole_generated | Fake | Fake | trufor → probe | [008__genimage__SD15__459_sdv5_00195__q95.html](008__genimage__SD15__459_sdv5_00195__q95.html) |
| 9 | `single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | probe_only | Fake | Fake | probe | [009__single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95.html](009__single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95.html) |
| 10 | `single_trufor__autosplice__54029__1__q95` | trufor_only | Fake | Fake | trufor | [010__single_trufor__autosplice__54029__1__q95.html](010__single_trufor__autosplice__54029__1__q95.html) |
| 11 | `autosplice__52769__authentic__q95` | trufor_mid_low_reliability | Real | Uncertain | trufor → probe | [011__autosplice__52769__authentic__q95.html](011__autosplice__52769__authentic__q95.html) |
| 12 | `autosplice__57332__authentic__q95` | trufor_mid_real | Real | Real | trufor → probe | [012__autosplice__57332__authentic__q95.html](012__autosplice__57332__authentic__q95.html) |
| 13 | `autosplice__67006__authentic__q95` | probe_high_real | Real | Uncertain | probe → trufor | [013__autosplice__67006__authentic__q95.html](013__autosplice__67006__authentic__q95.html) |
| 14 | `autosplice__70684__authentic__q95` | trufor_mid_real | Real | Uncertain | trufor → probe | [014__autosplice__70684__authentic__q95.html](014__autosplice__70684__authentic__q95.html) |
| 15 | `autosplice__45914__2__q95` | strong_local_evidence | Fake | Fake | probe → trufor | [015__autosplice__45914__2__q95.html](015__autosplice__45914__2__q95.html) |
| 16 | `autosplice__64223__2__q95` | strong_local_evidence | Fake | Fake | trufor → probe | [016__autosplice__64223__2__q95.html](016__autosplice__64223__2__q95.html) |
| 17 | `autosplice__72548__1__q95` | strong_local_evidence | Fake | Fake | trufor → probe | [017__autosplice__72548__1__q95.html](017__autosplice__72548__1__q95.html) |
| 18 | `genimage__ADM__921_adm_7__q95` | whole_generated | Fake | Fake | probe → trufor | [018__genimage__ADM__921_adm_7__q95.html](018__genimage__ADM__921_adm_7__q95.html) |
| 19 | `genimage__BigGAN__545_biggan_00074__q95` | whole_generated | Fake | Fake | probe → trufor | [019__genimage__BigGAN__545_biggan_00074__q95.html](019__genimage__BigGAN__545_biggan_00074__q95.html) |
| 20 | `genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | whole_generated | Fake | Fake | probe → trufor | [020__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95.html](020__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95.html) |
| 21 | `genimage__Midjourney__229_midjourney_197__q95` | whole_generated | Fake | Fake | probe → trufor | [021__genimage__Midjourney__229_midjourney_197__q95.html](021__genimage__Midjourney__229_midjourney_197__q95.html) |
| 22 | `genimage__Wukong__217_wukong_image88__q95` | whole_generated | Fake | Fake | probe → trufor | [022__genimage__Wukong__217_wukong_image88__q95.html](022__genimage__Wukong__217_wukong_image88__q95.html) |
| 23 | `genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95` | clean_real | Real | Real | probe → trufor | [023__genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95.html](023__genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95.html) |
| 24 | `genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95` | probe_high_real | Real | Uncertain | probe → trufor | [024__genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95.html](024__genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95.html) |
| 25 | `genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95` | trufor_borderline_real | Real | Real | trufor → probe | [025__genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95.html](025__genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95.html) |
| 26 | `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95` | trufor_mid_low_reliability | Real | Real | trufor → probe | [026__genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95.html](026__genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95.html) |

## 候选结论分布（审核用）

- 26 条全部完成；专家调用次数分布 [1, 2]；证据/覆盖/certainty 检查均为 pass。
- 总体与真值一致 21/26；整图维度 16/26；局部维度 22/26。这是审核对照，不是验收分数。
- 总体 Uncertain 共 5 条：`autosplice__65547__authentic__q95`、`autosplice__52769__authentic__q95`、`autosplice__67006__authentic__q95`、`autosplice__70684__authentic__q95`、`genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95`。
- 已确认判例：PROBE 高分 + 局部证据充分但整图来源无独立支撑 → 整图 Uncertain、局部 Tampered、总体 Fake；55310 因 PROBE 低分保留整图 Real 作对照。
- 已确认判例：TruFor 响应在整图边缘而非内容区 → 局部 Not tampered；51186、57332 按此口径。
- 65547、52769 保留局部 Uncertain；70684、67006 等仍按原候选保留。

## 审核时请重点判断

1. 每一步是否只引用了该轮之前已经投递的证据；有没有提前使用真值或 mask。
2. 整图来源与局部篡改是否分开写；PROBE 高分是否被错误写成整图生成。
3. TruFor 高分真图是否被写成 Tampered；边缘响应是否被当成内容异常。
4. 单专家样板是否真的“证据够就停”：PROBE-only 是否不需要局部；TruFor-only 是否把整图留 Uncertain。
5. 局部框是否与异常图/可靠性图所示区域一致；框过宽时 certainty 是否降级。

## 边界

- `REVIEW_QUESTIONS.md` 记录本轮已确认判例和复看点；`REVISION_NOTES.md`
  逐项列出改了标签、certainty 或调用理由的案例。
- 审核图片 `review_assets/`：逐条复制运行时的输入图、源图、实际投送的
  专家图和预测框叠加；单专家页按实际调用展示。页面改为批次内相对路径。
- 收尾审计 `AUDIT_REPORT.md`：26/26 pass，0 问题（证据引用时序、数字出处、
  真值/mask 词、页面结构、链接、划分与组成）。
- 候选文本是拟写稿，不是 Qwen 原始输出；运行记录明确标注 `sft_candidate_scripted`。
- 真值与 mask 只在审核侧；助手上下文没有它们。
- 本轮不扩样、不训练、不转 final_v2、不碰 test。
