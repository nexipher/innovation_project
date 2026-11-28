# 30 图验证：原版 Qwen3-VL vs LoRA

- 样本：30；local_tampered 10、真图 10、整图生成 0 条 both-correct（共 10）。
- 原版：局部检出 2/10；真图局部误报 0/10；两分项同时判对 8/30；弃权 0/30；调用 54（均 1.8）。
- LoRA：局部检出 1/10；真图局部误报 0/10；两分项同时判对 7/30；弃权 2/30；调用 49（均 1.633）。
- 运行：base 1488.7 s / peak 21.371 GB；LoRA 1862.0 s / peak 21.481 GB。

## 分项

| 指标 | 原版 | LoRA |
|---|---:|---:|
| 局部篡改检出 | 2/10 | 1/10 |
| 真图局部误报 | 0/10 | 0/10 |
| 真图整图误报（AI-generated） | 2 | 2 |
| 真图总体误报（Fake） | 2 | 2 |
| 两分项同时判对（30） | 8 | 7 |
| 其中篡改图（10） | 1 | 1 |
| 其中生成图（10） | 0 | 0 |
| 其中真图（10） | 7 | 6 |
| 弃权 | 0 | 2 |
| 专家调用总数 | 54 | 49 |
| 平均每会话调用 | 1.8 | 1.633 |

## 调用分布

- 原版：`{2: 24, 1: 6}`
- LoRA：`{2: 19, 1: 11}`

## 变化最大的案例

| sample | group | 原版 whole/local/overall | LoRA whole/local/overall | 原版/ LoRA both | calls |
|---|---|---|---|---|---|
| `val__genimage__Real__d3429f29-5d3a-4da3-829a-0fb387554a98__q95` | real | Real / Not tampered / Real | Real / Uncertain / Real | True / False | 2 / 1 |
| `val__autosplice__43686__1__q95` | local_tampered | AI-generated / Uncertain / Fake | AI-generated / Uncertain / Fake | False / False | 2 / 1 |
| `val__autosplice__52191__authentic__q95` | real | Real / Not tampered / Real | Real / Not tampered / Real | True / True | 2 / 1 |
| `val__autosplice__56086__0__q95` | local_tampered | AI-generated / Not tampered / Fake | AI-generated / Uncertain / Fake | False / False | 2 / 1 |
| `val__autosplice__57143__0__q95` | local_tampered | AI-generated / Tampered / Fake | AI-generated / Uncertain / Uncertain | False / False | 2 / 2 |
| `val__autosplice__73275__1__q95` | local_tampered | AI-generated / Uncertain / Fake | AI-generated / Uncertain / Fake | False / False | 1 / 2 |
| `val__genimage__BigGAN__739_biggan_00039__q95` | whole_generated | AI-generated / Not tampered / Fake | AI-generated / Uncertain / Fake | False / False | 2 / 1 |
| `val__genimage__Glide__GLIDE_1000_200_08_806_glide_00039__q95` | whole_generated | AI-generated / Uncertain / Fake | AI-generated / Uncertain / Fake | False / False | 2 / 1 |
| `val__genimage__Midjourney__487_midjourney_197__q95` | whole_generated | Uncertain / Tampered / Fake | Uncertain / Uncertain / Fake | False / False | 1 / 1 |
| `val__genimage__Real__d1b49378-c9ac-40d2-b520-d16ce16d3805__q95` | real | Real / Uncertain / Real | Real / Uncertain / Uncertain | False / False | 2 / 2 |
| `val__genimage__VQDM__VQDM_1000_200_01_143_vqdm_00143__q95` | whole_generated | AI-generated / Uncertain / Fake | AI-generated / Not tampered / Fake | False / False | 2 / 2 |

## 边界

- 同一 30 图、同一自主流程、同一 PROBE/TruFor；仅 Qwen 是否加载 LoRA 不同。
- 两分项同时判对 = 整图与局部都与任务真值一致；不是总体准确率。
- 真图误报按 truth 为 Real 的 10 条（5 AutoSplice 真图 + 5 Real）。
- 生成图的局部真值为 Not applicable；本表不把局部响应算作误报。
