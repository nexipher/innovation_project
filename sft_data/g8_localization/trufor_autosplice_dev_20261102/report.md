# TruFor AutoSplice q95 定位评测

装置：`cuda`；输入：`sft_data/g8_localization/autosplice_dev_subset_20261102.json`；检查 220/220，失败 0。

> mask 为 base 级共享；同源多变体按 base 聚合，不当作独立源。

| 项目 | 值 |
| --- | --- |
| forged variant IoU | mean 0.537，median 0.577 |
| forged variant F1 | mean 0.645，median 0.732 |
| forged source IoU | mean 0.523 |
| real high-response | mean 0.062，>1% sources 68/100 |
| image score AUROC (descriptive) | 0.8624 |
| source score AUROC | 0.8784 |

完整逐图指标、源级聚合、失败和资源字段见同目录 `report.json`。
