# 首轮 SFT 候选样板 — 收尾审计

- 审计样本：24/24；发现问题：0；全局问题：0。
- 审计不加载模型、不读 GPU；只读 manifest/split/records/traces/页面。
- 检查项：证据引用时序、数字与已投递 token 对齐、真值/mask 词、结论字段、
  页面原图/三图/分数表/调用顺序/结案/链接、划分与组成。

## 全局

- 无。

## 逐条

| # | sample | assistant turns | evidence | 结果 |
|---|---|---:|---:|---|
| 1 | `autosplice__54029__1__q95` | 3 | 2 | pass |
| 2 | `autosplice__55310__1__q95` | 3 | 2 | pass |
| 3 | `autosplice__45749__1__q95` | 3 | 2 | pass |
| 4 | `autosplice__69868__0__q95` | 3 | 2 | pass |
| 5 | `autosplice__48251__0__q95` | 3 | 2 | pass |
| 6 | `autosplice__51186__authentic__q95` | 3 | 2 | pass |
| 7 | `autosplice__65547__authentic__q95` | 3 | 2 | pass |
| 8 | `genimage__SD15__459_sdv5_00195__q95` | 3 | 2 | pass |
|  | `autosplice__52769__authentic__q95` | 3 | 2 | pass |
|  | `autosplice__57332__authentic__q95` | 3 | 2 | pass |
|  | `autosplice__67006__authentic__q95` | 3 | 2 | pass |
|  | `autosplice__70684__authentic__q95` | 3 | 2 | pass |
|  | `autosplice__45914__2__q95` | 3 | 2 | pass |
|  | `autosplice__64223__2__q95` | 3 | 2 | pass |
|  | `autosplice__72548__1__q95` | 3 | 2 | pass |
|  | `genimage__ADM__921_adm_7__q95` | 3 | 2 | pass |
|  | `genimage__BigGAN__545_biggan_00074__q95` | 3 | 2 | pass |
|  | `genimage__Glide__GLIDE_1000_200_01_185_glide_00020__q95` | 3 | 2 | pass |
|  | `genimage__Midjourney__229_midjourney_197__q95` | 3 | 2 | pass |
|  | `genimage__Wukong__217_wukong_image88__q95` | 3 | 2 | pass |
|  | `genimage__Real__96d68646-b0a2-450e-ac0a-a5a1b58aa6c8__q95` | 3 | 2 | pass |
|  | `genimage__Real__ac71afc5-10b2-4a79-86e2-aa55191c0d10__q95` | 3 | 2 | pass |
|  | `genimage__Real__b3cfabea-3cae-414b-a9e7-37f421542552__q95` | 3 | 2 | pass |
|  | `genimage__Real__e4ce4d06-00f7-456f-a17e-6049516d9668__q95` | 3 | 2 | pass |

## 审计边界

- 数字检查覆盖 0.x 形式；它验证“出处”，不验证文字解释是否最优。
- 真值/mask 词检查是词表检查；未命中不等于已证明没有泄露。
- 页面检查是结构检查；内容质量仍需人工审核。
- 本审计不改变任何候选轨迹，也不把候选变成训练数据。
