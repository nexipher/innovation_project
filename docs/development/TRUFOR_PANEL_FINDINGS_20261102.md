# TruFor 16 张 AutoSplice 面板检查（2026-11-02）

本文件基于已完成的 CPU 小批结果，**没有重跑模型**。四联图与 contact sheet：

- `sft_data/g8_localization/panels_20261102/contact_sheet.png`
- `sft_data/g8_localization/panels_20261102/01__*.png` … `16__*.png`
- 面板清单：`sft_data/g8_localization/panels_20261102/manifest.json`
- 原始指标：`sft_data/g8_localization/trufor_autosplice_smoke_20261102_cpu.json`

## 目视与定量结论

| 项目 | 观察 |
| --- | --- |
| Forged 覆盖 | 高响应几乎都在 mask 内（precision 0.980–1.000），但常只覆盖脸/颈/局部物体；GT mask 覆盖整个被操作对象时 recall 明显不足。 |
| Forged 数值 | IoU 均值 **0.307**、中位 **0.207**、范围 **0.076–0.641**；F1 均值 **0.434**；仅 2/8 IoU > 0.6。 |
| 典型欠覆盖 | 52550_0：map 只亮脸/颈，GT 覆盖整个头肩，IoU 0.128、recall 0.128；44699_1：IoU 0.194、recall 0.194；71174_0：IoU 0.184。 |
| 典型较好覆盖 | 64910_1：IoU 0.641、F1 0.781；71305_2：IoU 0.630、F1 0.773。 |
| 真图误报 | 8 张真图中 **4/8** 全图高响应面积 >1%；67206 达 **55.1%**（score 0.923），40269 16.3%（score 0.525），55518 19.9%（score 0.246）。 |
| score 与 map 不一致 | 55518 的 map 高响应面积 19.9%，但图像级 score 仅 0.246；不能再把“大面积高响应”直接等同于高图像分数。 |
| conf | 部分真图的 conf 也偏高（67206 conf mean 0.821、40269 0.679），不能仅靠 conf 抑制误报；52550_0 的 conf 大面积覆盖 mask，但 map 只覆盖局部。 |

## 逐张指标

| # | variant_id | label | score | IoU | F1 | recall | high-area |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `autosplice__52550__0__q95` | Fake | 0.736 | 0.128 | 0.228 | 0.128 | 0.088 |
| 2 | `autosplice__52550__1__q95` | Fake | 0.400 | 0.220 | 0.361 | 0.220 | 0.150 |
| 3 | `autosplice__52550__2__q95` | Fake | 0.189 | 0.076 | 0.142 | 0.076 | 0.052 |
| 4 | `autosplice__44699__1__q95` | Fake | 0.223 | 0.194 | 0.325 | 0.194 | 0.071 |
| 5 | `autosplice__64910__1__q95` | Fake | 0.514 | 0.641 | 0.781 | 0.648 | 0.216 |
| 6 | `autosplice__71174__0__q95` | Fake | 0.827 | 0.184 | 0.311 | 0.184 | 0.110 |
| 7 | `autosplice__40433__0__q95` | Fake | 0.620 | 0.382 | 0.553 | 0.382 | 0.196 |
| 8 | `autosplice__71305__2__q95` | Fake | 0.925 | 0.630 | 0.773 | 0.638 | 0.273 |
| 9 | `autosplice__49651__authentic__q95` | Real | 0.292 | — | — | — | 0.017 |
| 10 | `autosplice__43815__authentic__q95` | Real | 0.120 | — | — | — | 0.004 |
| 11 | `autosplice__67206__authentic__q95` | Real | 0.923 | — | — | — | 0.551 |
| 12 | `autosplice__60661__authentic__q95` | Real | 0.256 | — | — | — | 0.002 |
| 13 | `autosplice__40269__authentic__q95` | Real | 0.524 | — | — | — | 0.163 |
| 14 | `autosplice__52206__authentic__q95` | Real | 0.132 | — | — | — | 0.000 |
| 15 | `autosplice__55518__authentic__q95` | Real | 0.246 | — | — | — | 0.199 |
| 16 | `autosplice__47688__authentic__q95` | Real | 0.130 | — | — | — | 0.009 |

## 处置

- 这 16 张只用于管线/几何/误报检查，**不足以选定定位专家**；TruFor 的定位 recall 偏低、真图误报明显，需要扩大固定 dev 子集再评估。
- 已固定 AutoSplice q95 dev 子集：100 base / 220 输入（见 `AUTOSPLICE_LOCALIZATION_SUBSET_20261102.md`），当前无 GPU/2 GiB cgroup，尚未在子集上跑模型。
- 不改 Agent 区域观察、证据回传或结案流程；IML-ViT 代码/权重仍未完成核实。
