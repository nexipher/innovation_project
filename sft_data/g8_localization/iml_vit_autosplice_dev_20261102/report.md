# IML-ViT AutoSplice q95 定位评测

checkpoint `iml-vit_checkpoint_trufor_20231104.pth`（sha256 `9fa9ae88cafeb6ea…`）；device `cuda`；输入 `sft_data/g8_localization/autosplice_dev_subset_20261102.json`。

> IML-ViT **没有图像级 score/conf**；本报告只给 map 定位指标，不补造分数。mask 为 base 级共享，同源变体按 base 聚合。

| 项目 | 值 |
| --- | --- |
| forged variant IoU | mean 0.161，median 0.086 |
| forged variant F1 | mean 0.239，median 0.158 |
| forged source IoU | mean 0.160 |
| real high-response | mean 0.047，>1% sources 58/100 |
| score | 无（模型不提供） |

完整逐图指标、源级聚合、失败和资源字段见同目录 `report.json`。
