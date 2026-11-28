# 首轮 SFT 候选样板：重点案例清单

这 8 条按讨论价值排在审核工作台最前面；完整 24 条见 `index.html`。
助手文本是候选写手拟写，不是 Qwen 输出；PROBE/TruFor 与三图是真实运行。
真值和 mask 只用于审核，助手每一步只引用当时已收到的证据。

## 重点案例

| # | sample | 类型 | 真值 whole/local | 候选 whole/local/overall | PROBE | TruFor | 重点看什么 |
|---|---|---|---|---|---:|---:|---|
| 1 | `autosplice__54029__1__q95` | strong_local_evidence | Real / Tampered | Real / Tampered / Fake | 0.721901 | 0.999517 | 强局部证据；PROBE 中高分但局部高响应只在右侧白色衣物，示范整图与局部分开。 |
| 2 | `autosplice__55310__1__q95` | strong_local_evidence | Real / Tampered | Real / Tampered / Fake | 0.078204 | 0.999163 | PROBE 低分不能清除局部高响应；报纸矩形与高响应重合。 |
| 3 | `autosplice__45749__1__q95` | strong_local_evidence | Real / Tampered | Real / Tampered / Fake | 0.999415 | 0.998262 | PROBE 接近 1 但实际是局部编辑；示范不要把整图高分直接写成整图生成。 |
| 4 | `autosplice__69868__0__q95` | strong_local_evidence | Real / Tampered | Real / Tampered / Fake | 0.810086 | 0.995207 | 强局部证据、局部框较宽；适合讨论定位精度和 certainty。 |
| 5 | `autosplice__48251__0__q95` | strong_local_evidence | Real / Tampered | Real / Tampered / Fake | 0.930811 | 0.993394 | PROBE 高分的人像局部编辑；局部响应覆盖头部/肩部，框较宽。 |
| 6 | `autosplice__51186__authentic__q95` | trufor_high_real | Real / Not tampered | Real / Not tampered / Real | 0.000812 | 0.972335 | TruFor 0.97 但高响应只在画框右缘；示范读可靠性图后不写 Tampered。 |
| 7 | `autosplice__65547__authentic__q95` | trufor_high_real | Real / Not tampered | Real / Uncertain / Uncertain | 0.037187 | 0.897234 | TruFor 0.90 的真图；高响应在肩部/衣领，内容解释不唯一，保留 Uncertain。 |
| 8 | `genimage__SD15__459_sdv5_00195__q95` | whole_generated | AI-generated / Not applicable | AI-generated / Not applicable / Fake | 0.999973 | 0.979874 | 整图生成 + TruFor 0.98 局部高响应；示范整图 AI-generated 后局部 Not applicable。 |

## 全部 24 条

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
| 9 | `autosplice__52769__authentic__q95` | trufor_mid_low_reliability | Real | Uncertain | trufor → probe | [009__autosplice__52769__authentic__q95.html](009__autosplice__52769__authentic__q95.html) |
| 10 | `autosplice__57332__authentic__q95` | trufor_mid_real | Real | Uncertain | trufor → probe | [010__autosplice__57332__authentic__q95.html](010__autosplice__57332__authentic__q95.html) |
| 11 | `autosplice__67006__authentic__q95` | probe_high_real | Real | Uncertain | probe → trufor | [011__autosplice__67006__authentic__q95.html](011__autosplice__67006__authentic__q95.html) |
| 12 | `autosplice__70684__authentic__q95` | trufor_mid_real | Real | Uncertain | trufor → probe | [012__autosplice__70684__authentic__q95.html](012__autosplice__70684__authentic__q95.html) |
| 13 | `autosplice__45914__2__q95` | strong_local_evidence | Fake | Fake | probe → trufor | [013__autosplice__45914__2__q95.html](013__autosplice__45914__2__q95.html) |
| 14 | `autosplice__64223__2__q95` | strong_local_evidence | Fake | Fake | trufor → probe | [014__autosplice__64223__2__q95.html](014__autosplice__64223__2__q95.html) |
| 15 | `autosplice__72548__1__q95` | strong_local_evidence | Fake | Fake | trufor → probe | [015__autosplice__72548__1__q95.html](015__autosplice__72548__1__q95.html) |
| 16 | `genimage__ADM__921_adm_7__q95` | whole_generated | Fake | Fake | probe → trufor | [016__genimage__ADM__921_adm_7__q95.html](016__genimage__ADM__921_adm_7__q95.html) |
| 17 | `genimage__BigGAN__545_biggan_00074__q95` | whole_generated | Fake | Fake | probe → trufor | [017__genimage__BigGAN__545_biggan_00074__q95.html](017__genimage__BigGAN__545_biggan_00074__q95.html) |
| 18 | `genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | whole_generated | Fake | Fake | probe → trufor | [018__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95.html](018__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95.html) |
| 19 | `genimage__Midjourney__229_midjourney_197__q95` | whole_generated | Fake | Fake | probe → trufor | [019__genimage__Midjourney__229_midjourney_197__q95.html](019__genimage__Midjourney__229_midjourney_197__q95.html) |
| 20 | `genimage__Wukong__217_wukong_image88__q95` | whole_generated | Fake | Fake | probe → trufor | [020__genimage__Wukong__217_wukong_image88__q95.html](020__genimage__Wukong__217_wukong_image88__q95.html) |
| 21 | `genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95` | clean_real | Real | Real | probe → trufor | [021__genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95.html](021__genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95.html) |
| 22 | `genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95` | probe_high_real | Real | Uncertain | probe → trufor | [022__genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95.html](022__genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95.html) |
| 23 | `genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95` | trufor_borderline_real | Real | Real | trufor → probe | [023__genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95.html](023__genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95.html) |
| 24 | `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95` | trufor_mid_low_reliability | Real | Real | trufor → probe | [024__genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95.html](024__genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95.html) |

## 候选结论分布（审核用）

- 24 条全部完成；每会话 2 次专家调用、3 轮助手输出；证据/覆盖/certainty 检查均为 pass。
- 总体与真值一致 18/24；整图维度 22/24；局部维度 19/24。这是审核对照，不是验收分数。
- 总体 Uncertain 共 6 条：`autosplice__65547__authentic__q95`、`autosplice__52769__authentic__q95`、`autosplice__57332__authentic__q95`、`autosplice__67006__authentic__q95`、`autosplice__70684__authentic__q95`、`genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95`。
- 其中 5 条 AutoSplice 真图和 1 条 Real 是有意选入的 TruFor 高分/边界/低可靠讨论案例；候选按协议把“分数高于阈值但只有候选”或“局部图低可靠”写成 Uncertain，未硬写 Not tampered。请审核这些案例该维持 Uncertain，还是应判 Real/Not tampered。

## 审核时请重点判断

1. 每一步是否只引用了该轮之前已经投递的证据；有没有提前使用真值或 mask。
2. 整图来源与局部篡改是否分开写；PROBE 高分是否被错误写成整图生成。
3. TruFor 高分真图是否被写成 Tampered；可靠性图低的区域是否被当成证据。
4. 补测时机是否合理：是否在整图或局部仍未解决时才补测；有没有为凑双调用而补测，或该补测时提前结案。
5. 局部框是否与异常图/可靠性图所示区域一致；框过宽时 certainty 是否降级。

## 边界

- 待定判例 `REVIEW_QUESTIONS.md`：TruFor 高分真图何时写 Not tampered、
  PROBE 高分的局部编辑整图维度怎么写、调用顺序/是否保留单专家样板。
- 审核图片 `review_assets/`：逐条复制运行时的输入图、源图、TruFor 三图、
  PROBE 裁剪和预测框叠加；页面改为批次内相对路径，拉取仓库后可直接打开。
- 收尾审计 `AUDIT_REPORT.md`：24/24 pass，0 问题（证据引用时序、数字出处、
  真值/mask 词、页面结构、链接、划分与组成）。
- 文字补丁 `POST_RUN_PATCHES.md`：Midjourney 229 的推理去掉了未投递的
  0.39 数字，改为定性读可靠性图；不影响工具输出、证据 ID、框或结论。
- 候选文本是拟写稿，不是 Qwen 原始输出；运行记录明确标注 `sft_candidate_scripted`。
- 真值与 mask 只在审核侧；助手上下文没有它们。
- 本轮不扩样、不训练、不转 final_v2、不碰 test。
