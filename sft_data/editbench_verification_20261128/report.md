# EditBench 生成图局部编辑核验

官方 EditBench 包是 379 MB tar；本轮未下载整包，只从公开 bucket 取 5 组单样本（reference + mask + 实际 Imagen Editor 输出），raw 约 14 MB、preview 约 1.1 MB，新增磁盘约 15 MB。

## 官方文件清单

- 官方页面：https://imagen.research.google/editor/；整包 `editbench.tar.gz` 页面标 379 MB（bucket 对象 397,810,521 B）。
- 公开 bucket 共 12,003 个对象、11,470,153,777 B；生成子集 120 个 item，`reference_generated` 与 `mask_generated` 各 120 个；`im_*`/`dl2_*`/`sd_*`/`imrm_*`是各编辑器/提示类型的输出；`annotations_generated.csv` 120 行。
- 本轮只取 5 组：5 个 reference、5 个 mask、5 个 `im_generated_mask-rich_1` 输出；未下载整包，也未下载 `dl2_*`/`sd_*`。

## 结论

**可以用于“AI 生成图随后被局部编辑”的评测，但需使用生成子集的 reference/mask/`im_generated_mask-rich_1` 三件套，不能把 reference 当成编辑后图。**
5/5 组满足：mask 外差异均值 <1 级，>5 级像素 ≤0.32%；mask 内平均差异 50–124，>10 级像素 66–91%；变化区域与 mask 的 IoU 0.66–0.91。

## 每组的角色

- `reference_generated/<stem>.png`：完整编辑前生成底图（1024×1024 RGBA），同时是数据集用于评分的 reference；不是编辑器输出。预览中底图水印有 Imagen 和 Parti 两种，说明生成子集的底图生成器混合，不全是 Imagen。
- `mask_generated/<stem>.png`：1024×1024 二值 mask，白色为编辑区域。
- `im_generated_mask-rich_1/<stem>.png`：实际 Imagen Editor 输出，1024×1024 RGB；`dl2_*`、`sd_*` 是同 benchmark 的其他编辑器输出，本轮未用。
- bucket 没有单独的 masked-input 文件；reference + mask 可重建输入，但官方未单独发布该中间图。

## 样本

| sample | row | 类别 | mask 占比 | 外均值 | 外>5 | 内均值 | 内>10 | 变化/mask IoU |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `eb_gen_029_pig` | 29 | color/common/outdoor | 0.354 | 0.71 | 0.21% | 50.3 | 81.5% | 0.814 |
| `eb_gen_025_dog` | 25 | color/common/painting | 0.377 | 0.71 | 0.17% | 106.0 | 66.0% | 0.660 |
| `eb_gen_015_sawfish` | 15 | material/uncommon/indoor | 0.169 | 0.58 | 0.13% | 107.5 | 88.9% | 0.886 |
| `eb_gen_064_triangle` | 64 | shape/writing/real | 0.203 | 0.78 | 0.21% | 82.3 | 74.8% | 0.745 |
| `eb_gen_114_three` | 114 | count/writing/painting | 0.330 | 0.93 | 0.31% | 123.9 | 91.2% | 0.909 |

## 边界

- 非编辑区不是逐位相同：5 组 mask 外仍有 0.13–0.31% 像素差异 >5 级，均值 <1 级，可能是输出重编码、RGBA→RGB、边缘/水印造成。若要求“mask 外必须严格 bit-exact”，EditBench 输出不满足，需要另行处理。
- 本次只核验 Imagen Editor 的 mask-rich 输出；`dl2_*`/`sd_*` 也可作为其他编辑器输出，但未下载。
- 预览在 `review/index.html`；逐样本 URL/指标在 `manifest.json` 和 `sources.csv`。
- 未跑 TruFor/Agent，未改规则，未下载整包。
