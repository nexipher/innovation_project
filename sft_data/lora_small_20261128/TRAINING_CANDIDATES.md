# 24 条 LoRA 小试训练样板 — 待确认清单

状态：**pending confirmation**。尚未启动 GPU 训练。标签以
`sft_data/sft_candidates_20261127/records.json` 为准；统一按
**整图 / 局部 / 总体** 展示。

## 分布

| 分布 | Fake | Real | Uncertain |
|---|---:|---:|---:|
| 真值 | 16 | 8 | 0 |
| 训练目标（总体） | 16 | 5 | 3 |

分项目标分布：整图 `Real 9 / Uncertain 8 / AI-generated 7`；
局部 `Tampered 9 / Not tampered 5 / Uncertain 3 / Not applicable 7`。

## 已通过、直接复用（3）

| sample | 整图 / 局部 / 总体 | 真值 |
|---|---|---|
| `autosplice__55310__1__q95` | Real / Tampered / Fake | Fake |
| `autosplice__65547__authentic__q95` | Real / Uncertain / Uncertain | Real |
| `single_trufor__autosplice__54029__1__q95` | Uncertain / Tampered / Fake | Fake |

## 待确认（21）

| # | sample | 类型 | 真值 | 整图 / 局部 / 总体 | calls | tokens | images |
|---:|---|---|---|---|---:|---:|---:|
| 1 | `autosplice__54029__1__q95` | strong_local_evidence | Fake | Uncertain / Tampered / Fake | 2 | 8972 | 5 |
| 2 | `autosplice__45749__1__q95` | strong_local_evidence | Fake | Uncertain / Tampered / Fake | 2 | 8837 | 5 |
| 3 | `autosplice__69868__0__q95` | strong_local_evidence | Fake | Uncertain / Tampered / Fake | 2 | 9005 | 5 |
| 4 | `autosplice__48251__0__q95` | strong_local_evidence | Fake | Uncertain / Tampered / Fake | 2 | 8958 | 5 |
| 5 | `autosplice__51186__authentic__q95` | trufor_high_real | Real | Real / Not tampered / Real | 2 | 8917 | 5 |
| 6 | `genimage__SD15__459_sdv5_00195__q95` | whole_generated | Fake | AI-generated / Not applicable / Fake | 2 | 9608 | 5 |
| 7 | `single_probe__genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | probe_only | Fake | AI-generated / Not applicable / Fake | 1 | 6626 | 2 |
| 8 | `autosplice__52769__authentic__q95` | trufor_mid_low_reliability | Real | Real / Uncertain / Uncertain | 2 | 8800 | 5 |
| 9 | `autosplice__57332__authentic__q95` | trufor_mid_real | Real | Real / Not tampered / Real | 2 | 8874 | 5 |
| 10 | `autosplice__70684__authentic__q95` | trufor_mid_real | Real | Real / Uncertain / Uncertain | 2 | 8871 | 5 |
| 11 | `autosplice__45914__2__q95` | strong_local_evidence | Fake | Uncertain / Tampered / Fake | 2 | 8879 | 5 |
| 12 | `autosplice__64223__2__q95` | strong_local_evidence | Fake | Uncertain / Tampered / Fake | 2 | 9057 | 5 |
| 13 | `autosplice__72548__1__q95` | strong_local_evidence | Fake | Uncertain / Tampered / Fake | 2 | 8854 | 5 |
| 14 | `genimage__ADM__921_adm_7__q95` | whole_generated | Fake | AI-generated / Not applicable / Fake | 2 | 8727 | 5 |
| 15 | `genimage__BigGAN__545_biggan_00074__q95` | whole_generated | Fake | AI-generated / Not applicable / Fake | 2 | 8669 | 5 |
| 16 | `genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | whole_generated | Fake | AI-generated / Not applicable / Fake | 2 | 8757 | 5 |
| 17 | `genimage__Midjourney__229_midjourney_197__q95` | whole_generated | Fake | AI-generated / Not applicable / Fake | 2 | 11677 | 5 |
| 18 | `genimage__Wukong__217_wukong_image88__q95` | whole_generated | Fake | AI-generated / Not applicable / Fake | 2 | 9171 | 5 |
| 19 | `genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95` | clean_real | Real | Real / Not tampered / Real | 2 | 8892 | 5 |
| 20 | `genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95` | trufor_borderline_real | Real | Real / Not tampered / Real | 2 | 9055 | 5 |
| 21 | `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95` | trufor_mid_low_reliability | Real | Real / Not tampered / Real | 2 | 9420 | 5 |

## 建议排除的 2 条

| sample | 原因 |
|---|---|
| `autosplice__67006__authentic__q95` | whole/local/overall 全 Uncertain，训练目标最弱 |
| `genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95` | overall Uncertain、whole Uncertain，局部无 Tampered |

## Dry-run 摘要

- 24/24 通过；token 范围 6626–11677；图片数范围 2–5。
- **最长且图最多**：`genimage__Midjourney__229_midjourney_197__q95`，11677 tokens，5 图。
- token 前 3：`genimage__Midjourney__229_midjourney_197__q95`, `genimage__SD15__459_sdv5_00195__q95`, `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95`。
- 图片数前 3（5 图并列，按 token 打破平局）：`genimage__Midjourney__229_midjourney_197__q95`, `genimage__SD15__459_sdv5_00195__q95`, `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95`。
- 10 步 profiling 改用：`genimage__Midjourney__229_midjourney_197__q95`, `genimage__SD15__459_sdv5_00195__q95`, `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95`。
- **max_length 警告**：1 条超过当前 10000：`genimage__Midjourney__229_midjourney_197__q95`。正式 profiling/训练前需把 max_length 提到至少 12000，或先决定截断策略；本次未改训练参数。

完整逐条 dry-run 见 `DRY_RUN.md`；原始 JSON 见 `dry_run.json`。

## 确认后执行顺序

1. 用上述 profiling 3 条跑 10 步，记录单步耗时、峰值显存；若 Midjourney 的
   11677 token 在 max_length 12000 下 OOM，先报告。
2. 从原版 Qwen3-VL 重新开始，24 条 × 2 epoch（48 步）正式训练并保存 adapter。
3. 约 30 张与 22 个训练源不同源的验证图，同一自主调用流程对比原版 vs LoRA；
   汇报局部检出、真图误报、两分项同时判对、弃权、调用次数和最大变化案例。
4. 不扩 250 条；OOM 先报告。
