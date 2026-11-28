# 24 条 LoRA 小试训练样板 — 待确认清单

状态：**pending confirmation**。尚未启动训练。

仓库里没有 `sft_candidates_20261127` 的候选审核决定文件；下面把上一轮明确用于
LoRA 冒烟的 3 条视为“已通过、直接复用”，其余 21 条列为待确认。若你已通过其他渠道
批准了别的轨迹，请指出。

## 已通过、直接复用（3）

| sample | 用途 | 标签 |
|---|---|---|
| `autosplice__55310__1__q95` | 报纸；TruFor→PROBE；整图 Real 对照 | Fake / Real / Tampered |
| `autosplice__65547__authentic__q95` | 肩部；TruFor→PROBE；TruFor 高分真图 | Real / Real / Uncertain |
| `single_trufor__autosplice__54029__1__q95` | 单专家 TruFor-only；整图 Uncertain | Fake / Uncertain / Tampered |

## 待确认（21）

| # | sample | 类型 | 候选 whole/local/overall | calls |
|---:|---|---|---|---:|
| 1 | `autosplice__54029__1__q95` | 晾衣绳；强局部 | Real→Uncertain / Tampered / Fake | 2 |
| 2 | `autosplice__45749__1__q95` | 企鹅；PROBE 高、局部强 | Uncertain / Tampered / Fake | 2 |
| 3 | `autosplice__69868__0__q95` | 老虎；强局部、框宽 | Uncertain / Tampered / Fake | 2 |
| 4 | `autosplice__48251__0__q95` | 人像；强局部、框宽 | Uncertain / Tampered / Fake | 2 |
| 5 | `autosplice__51186__authentic__q95` | 乌鸦；TruFor 高分、边缘响应 | Real / Not tampered / Real | 2 |
| 6 | `genimage__SD15__459_sdv5_00195__q95` | 整图生成 + TruFor 局部高响应 | AI-generated / Not applicable / Fake | 2 |
| 7 | `single_probe__genimage__Glide__...` | 单专家 PROBE-only | AI-generated / Not applicable / Fake | 1 |
| 8 | `autosplice__52769__authentic__q95` | 面部；低可靠 Uncertain | Real / Real / Uncertain | 2 |
| 9 | `autosplice__57332__authentic__q95` | 枪械；整图底边响应 | Real / Real / Real | 2 |
| 10 | `autosplice__70684__authentic__q95` | 北极熊；内容区轻微响应 | Real / Real / Uncertain | 2 |
| 11 | `autosplice__45914__2__q95` | 公交车；强局部 | Uncertain / Tampered / Fake | 2 |
| 12 | `autosplice__64223__2__q95` | 地面物件；强局部 | Uncertain / Tampered / Fake | 2 |
| 13 | `autosplice__72548__1__q95` | 舞者；强局部、定位宽 | Uncertain / Tampered / Fake | 2 |
| 14 | `genimage__ADM__921_adm_7__q95` | 整图生成 | AI-generated / Not applicable / Fake | 2 |
| 15 | `genimage__BigGAN__545_biggan_00074__q95` | 整图生成 | AI-generated / Not applicable / Fake | 2 |
| 16 | `genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | 整图生成 | AI-generated / Not applicable / Fake | 2 |
| 17 | `genimage__Midjourney__229_midjourney_197__q95` | 整图生成；局部图低可靠 | AI-generated / Not applicable / Fake | 2 |
| 18 | `genimage__Wukong__217_wukong_image88__q95` | 整图生成 | AI-generated / Not applicable / Fake | 2 |
| 19 | `genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95` | 干净 Real | Real / Not tampered / Real | 2 |
| 20 | `genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95` | 瀑布；TruFor 边界 | Real / Not tampered / Real | 2 |
| 21 | `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95` | 狐猴；局部图低可靠 | Real / Not tampered / Real | 2 |

## 建议排除的 2 条

| sample | 原因 |
|---|---|
| `autosplice__67006__authentic__q95` | whole/local/overall 全 Uncertain；训练目标最弱 |
| `genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95` | overall/whole Uncertain，局部无 Tampered；目标弱 |

## 组成

- 24 条 = 3 已通过 + 21 待确认。
- 按图像来源：AutoSplice 篡改 8 条（含 54029 两次）、AutoSplice 真图 5 条、
  GenImage 生成 6 条、GenImage Real 3 条、单专家 2 条。
- 标签：Fake 16、Real 8（含 Uncertain 目标 3 条：65547、52769、70684）。
- 22 个唯一源图；`single_trufor__54029` 与 `54029_1` 同源，
  `single_probe__Glide` 与 `genimage__Glide` 同源。验证集必须避开这 22 个源。

## 确认后执行顺序

1. 用已通过的 3 条（最长 65547：9019 token、5 图；55310：8853/5；54029：7669/4）
   先跑 10 步 profiling，记录单步耗时、峰值显存，估算 250 条 × 2 epoch。
2. 从原版 Qwen3-VL 重新开始，24 条 × 2 epoch（48 步），保存正式 adapter。
3. 约 30 张验证图：与 22 个训练源不同源，按同一自主调用流程对比原版 vs LoRA；
   汇报局部篡改检出、真图误报、两分项同时判对、弃权、调用次数和变化最大案例。
4. 不扩 250 条；OOM 先报告。
