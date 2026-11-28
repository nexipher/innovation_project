# EditBench 5 对 — 独立 TruFor

5 对、10 张图：reference 与 `im_generated_mask-rich_1` 各自从 PNG 转成 JPEG q95（quality=95, subsampling=2），TruFor 只读 JPEG；mask 只用于评估定位。
阈值 score >= 0.5 记检出；IoU >= 0.1 记定位命中。

## 聚合

| 指标 | 结果 |
|---|---:|
| 编辑图检出（score>=0.5） | 1/5 |
| 编辑图定位命中（IoU>=0.1） | 4/5 |
| 编辑图检出且定位 | 1/5 |
| 编辑图 mask 高响应>=0.1 | 4/5 |
| 原图 score 误报 | 1/5 |
| 原图 score+定位误报 | 0/5 |
| 原图 mask IoU>=0.1（未计 score） | 1/5 |
| 原图 mask 高响应>=0.1（未计 score） | 1/5 |

## 逐对

| pair | 原 score | 编 score | Δscore | 原 mask mean | 编 mask mean | 编 IoU | 编高响应 | 检出/定位 | 低分但热图覆盖 |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `eb_gen_029_pig` | 0.160 | 0.115 | -0.045 | 0.038 | 0.276 | 0.187 | 0.191 | N/Y | Y |
| `eb_gen_025_dog` | 0.275 | 0.473 | +0.198 | 0.031 | 0.016 | 0.000 | 0.000 | N/N | N |
| `eb_gen_015_sawfish` | 0.947 | 0.601 | -0.346 | 0.000 | 0.593 | 0.601 | 0.622 | Y/Y | N |
| `eb_gen_064_triangle` | 0.152 | 0.469 | +0.316 | 0.153 | 0.332 | 0.293 | 0.293 | N/Y | Y |
| `eb_gen_114_three` | 0.265 | 0.336 | +0.070 | 0.576 | 0.515 | 0.420 | 0.583 | N/Y | Y |

## 图级分数低但热图覆盖编辑区

`eb_gen_029_pig`, `eb_gen_064_triangle`, `eb_gen_114_three`

逐例：

- `eb_gen_029_pig`：score 0.115（<0.5），IoU 0.187、mask 高响应 0.191；原图同区域高响应 0.0004，属于编辑后新出现的局部响应。
- `eb_gen_064_triangle`：score 0.469（刚低于 0.5），IoU 0.293、高响应 0.293；原图 0.072，主要是编辑后响应。
- `eb_gen_114_three`：score 0.336，IoU 0.420、高响应 0.583；但原图同区域已有高响应 0.788、IoU 0.537，编辑后反而下降，不能把热图覆盖当作本次编辑证据。

其他：`eb_gen_025_dog` score 0.473（接近阈值）但 IoU 0、mask 高响应 0，属于 TruFor 未定位；`eb_gen_015_sawfish` score 0.601 检出，但原图 score 0.947 是图级误报（原图热图并未覆盖 mask）。

## 代表热图

`previews/<pair_id>.jpg` 为 reference / actual output / mask overlay / TruFor anomaly map 四栏；完整 5 对预览见 `previews/`。

## 产物

- `per_pair.csv`：5 行逐对指标；`per_image.csv`：10 行逐图指标。
- `trufor_analysis.json`：完整指标、阈值与编码参数。
- `trufor_maps/`：每张图的 anomaly/confidence 图；`previews/`：代表图。
- 输入 JPEG 在 `inputs/`；原始 PNG 只在 `/tmp/editbench_samples`。
- `trufor_raw/` 的 81 MB npz 缓存核验后未保留；需要重算时重跑 10 张即可。

## 边界

本轮只跑独立 TruFor；未跑 Agent，未安装 PIM，未改规则/训练。EditBench mask 外不是逐位 bit-exact，详见上一轮核验报告。
