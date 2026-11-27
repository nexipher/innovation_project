# 独立专家对照：PROBE / TruFor × 200 q95

输入：`sft_data/g8_dev_eval_20261110/manifest.json`（200 张，原顺序）；运行时间：2026-11-27T12:50:50+0800 – 2026-11-27T12:53:24+0800。

## 口径

- PROBE：`raw_metric`，阈值 **0.5**（外部默认；未校准概率，仅作判别分数）。
- TruFor 图像级：score 阈值 **0.5**；map 高响应阈值 **0.5**。
- AutoSplice 定位：base 共享 mask，resize 到 map；IoU/F1/precision/recall 按像素计算；本批每个 base 只取一张，不把同源变体当独立样本。
- 独立专家不加载 Qwen、不经过 Agent；Agent 记录来自 2026-11-27 两批重跑。

## PROBE

| 组 | n | score≥0.5 | 比率 | 说明 |
|---|---:|---:|---:|---|
| GenImage 生成图 | 80 | 74 | 0.925 | 检出率 |
| GenImage 真图 | 30 | 2 | 0.0667 | 误报率 |
| AutoSplice 真图 | 30 | 8 | 0.2667 | 整图误报率 |
| AutoSplice 编辑图 | 60 | 43 | 0.7167 | 整图误报率（真值整图 Real） |

GenImage 分组：

| 生成器 | n | 检出 | 检出率 |
|---|---:|---:|---:|
| ADM | 10 | 10 | 1.0 |
| BigGAN | 10 | 8 | 0.8 |
| Glide | 10 | 6 | 0.6 |
| Midjourney | 10 | 10 | 1.0 |
| SD14 | 10 | 10 | 1.0 |
| SD15 | 10 | 10 | 1.0 |
| VQDM | 10 | 10 | 1.0 |
| Wukong | 10 | 10 | 1.0 |

## TruFor

| 指标 | 值 |
|---|---|
| AutoSplice 篡改图像级检出 | 41/60 = 0.6833 |
| AutoSplice 真图误报 | 6/30 = 0.2000 |
| 图像级 TP/FP/FN | 41/6/19 |
| 图像级 P/R/F1 | 0.8723 / 0.6833 / 0.7664 |
| 定位 IoU mean / median | 0.5303 / 0.5422 |
| 定位 F1 mean / median | 0.6445 / 0.7032 |
| 定位 IoU≥0.1 / 0.3 / 0.5 | 55/47/33（n=60） |
| GenImage 80 生成图 score≥0.5 | 11/80（局部响应单列，不是误报） |
| GenImage 80 map high fraction mean | 0.0738 |
| GenImage 30 真图 score≥0.5 | 6/30 |
| GenImage 30 map high fraction mean | 0.0928 |

阈值：TruFor 图像 score **0.5**；map 高响应 **0.5**；AutoSplice mask 为 base 共享 mask，resize 到 map；本批每个 base 只取一张。

## Agent 重点组

| 组 | n | 独立 TruFor score 检出 | 检出率 | 定位 IoU≥0.1 | 定位率 |
|---|---:|---:|---:|---:|---:|
| tampered_agent_not_called | 14 | 10 | 0.7143 | 13 | 0.9286 |
| tampered_agent_called_not_detected | 33 | 18 | 0.5455 | 29 | 0.8788 |
| tampered_agent_detected | 13 | 13 | 1.0 | 13 | 1.0 |

Agent 已调用且有效的 TruFor score 与本次独立重跑逐图一致：146/146（0 条差异）。

问题层级读数（描述性，不是自动验收）：

- `tampered_agent_not_called`：{'tool_call': 10, 'agent_interpretation': 0, 'expert_output': 0, 'localization_only': 4}。`tool_call` 表示独立 TruFor score≥0.5、Agent 本可调用却未调用；`localization_only` 表示 score<0.5 但 IoU≥0.1，定位已有候选。
- `tampered_agent_called_not_detected`：{'tool_call': 0, 'agent_interpretation': 18, 'expert_output': 4, 'localization_only': 11}。`agent_interpretation` 表示独立 score≥0.5、Agent 已调用但未写 local Tampered；`expert_output` 表示独立 score<0.5 且无定位命中；`localization_only` 为边界情况。

读法：14 张未调用中 10 张独立 score≥0.5、13 张 IoU≥0.1，问题首先在**调用覆盖**；33 张已调用未检出中 18 张独立 score≥0.5，问题更像**Agent 判读/证据使用**；4 张独立 score<0.5 且无定位命中，属于**专家输出限制**；11 张仅定位命中，需人工看响应区域是否被忽略。

### tampered_agent_not_called 代表案例

| sample_id | Agent local | TruFor score | IoU | focus |
|---|---|---:|---:|---|
| autosplice__61270__2__q95 | Uncertain | 0.999578 | 0.6749947949198417 | tampered_agent_not_called |
| autosplice__44492__0__q95 | Uncertain | 0.997316 | 0.5370233199662844 | tampered_agent_not_called |
| autosplice__73258__0__q95 | Uncertain | 0.959324 | 0.5996333380341278 | tampered_agent_not_called |
| autosplice__69769__2__q95 | Uncertain | 0.900236 | 0.8105096330048042 | tampered_agent_not_called |
| autosplice__45563__2__q95 | Uncertain | 0.893564 | 0.48741951787637067 | tampered_agent_not_called |

### tampered_agent_called_not_detected 代表案例

| sample_id | Agent local | TruFor score | IoU | focus |
|---|---|---:|---:|---|
| autosplice__41242__1__q95 | Uncertain | 0.995490 | 0.8117188862407719 | tampered_agent_called_not_detected |
| autosplice__53560__0__q95 | Uncertain | 0.985356 | 0.8325557866336912 | tampered_agent_called_not_detected |
| autosplice__42911__1__q95 | Uncertain | 0.985017 | 0.5168772392985103 | tampered_agent_called_not_detected |
| autosplice__72315__1__q95 | Uncertain | 0.970479 | 0.9130472535296876 | tampered_agent_called_not_detected |
| autosplice__68092__2__q95 | Uncertain | 0.965022 | 0.8434947700243588 | tampered_agent_called_not_detected |

### tampered_agent_detected 代表案例

| sample_id | Agent local | TruFor score | IoU | focus |
|---|---|---:|---:|---|
| autosplice__63103__1__q95 | Tampered | 0.999402 | 0.882389289392379 | tampered_agent_detected |
| autosplice__67217__0__q95 | Tampered | 0.999173 | 0.6862546437014522 | tampered_agent_detected |
| autosplice__50949__2__q95 | Tampered | 0.997696 | 0.7923797223119148 | tampered_agent_detected |
| autosplice__43680__1__q95 | Tampered | 0.997455 | 0.7329749103942652 | tampered_agent_detected |
| autosplice__62791__0__q95 | Tampered | 0.987046 | 0.8482709191901093 | tampered_agent_detected |

## 边界

- 独立专家只回答自己的测量问题；PROBE 无定位，TruFor 的 score 不是本项目校准概率。
- GenImage 生成图的局部真值为 Not applicable，局部响应单列，不计入 AutoSplice 误报。
- 逐图 CSV：`per_sample.csv`；TruFor 原始 npz：`trufor_raw/`；异常/可靠性图：`trufor_maps/`。
