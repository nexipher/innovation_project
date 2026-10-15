# v1 归档说明（legacy_v1_20261015）

2026-10-15T09:14:43 归档。**这份归档是历史实验，不是训练材料**：全部批次已冻结
（`sft_data/FROZEN.json`），训练入口遇到即拒绝。

- 清单：[`MANIFEST.json`](MANIFEST.json)（每一项的原路径、大小、摘要）
- 路径映射：[`../PATH_MAP.json`](../PATH_MAP.json)（旧路径 → 归档路径）
- 留在原位的运行时资产见 MANIFEST 的 `stays_in_place`

**使用归档时的三条规矩**

1. 复算请走 `utils.batch.resolve_path()`：命中归档副本时，报告里必须写
   `resolved_from_archive=true`——"读的是副本"是结果的一部分。
2. 归档页面的图片链接已按映射改写（0 个 HTML 文件）；**原页面未改**。
3. 归档不解除门禁：`apply_review` / `build_final_v2` / 轨迹生成对冻结批次仍然拒绝。
