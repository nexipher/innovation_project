# TruFor vs IML-ViT 定位对照（2026-11-05）

同一批 **100 base / 220 输入**（100 Real + 120 Fake；mask base 级共享）。完整数据：`sft_data/g8_localization/localization_comparison_20261105.json`；16 张目视对照：`sft_data/g8_localization/localization_compare16_20261105/contact_sheet.png`。

## 总体

| 指标 | TruFor | IML-ViT |
| --- | ---: | ---: |
| forged variant IoU mean | 0.537 | 0.161 |
| forged variant F1 mean | 0.645 | 0.239 |
| forged variant recall mean | 0.570 | 0.176 |
| forged source IoU mean | 0.523 | 0.160 |
| real high-area mean | 0.062 | 0.047 |
| real >1% sources | 68/100 | 58/100 |
| source score AUROC | 0.878 [0.824, 0.925] | 无 score |

## 同源配对

- 70 个 forged base 中，TruFor 的 source IoU 更高 **61** 个，IML 更高 **9** 个；Δ(TruFor−IML) 均值 **0.363**、中位 **0.404**。
- 100 个真实源中，TruFor 高响应面积更高 53 个，IML 更高 47 个；均值 TruFor **0.062** vs IML **0.047**。

## 目视

- TruFor 的 map 覆盖率明显更高；IML-ViT 的 map 常集中在边缘/局部高响应，precision 高但 recall 低（例如 64910_1：TruFor IoU 0.641，IML IoU 0.044）。
- IML 也有少数 source 优于 TruFor（例如 72093：IML source IoU 0.925 vs TruFor 0.000），所以不能把结论写成“逐图全胜”；但按源大多数更差。

## 资源与结论

- TruFor：冷跑 **107.67 s**，0.489 s/张，7 个 batch；官方 subprocess 未提供可直接读取的 GPU peak。
- IML-ViT：**23.64 s**，load 3.817 s，0.107 s/张；GPU peak 9.9 / 10.941 GB。
- **初步结论**：在这一个固定 dev 子集、固定 0.5 map 阈值下，TruFor 的定位覆盖和 source IoU/F1 明显优于 IML-ViT；IML-ViT 只有稀疏高响应，且无图像级 score。IML-ViT 不作为当前定位专家候选；TruFor 作为初步定位专家候选。
- 限制：单子集、共享 base mask、无逐变体 mask；test 仍无未见性，不用于选择；不调阈值、不改 Agent。
