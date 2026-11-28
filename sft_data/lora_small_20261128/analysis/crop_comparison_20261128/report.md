# TruFor 裁剪窗口 A/B/C 离线对照

A = 当前 high-response 包围盒（含旧 peak 回退）；B = high-response 像素连通分量按 `mean anomaly × mean confidence` 排序的 Top-1；C = 同排序 Top-2 的两框并集。窗口都加 5% 上下文；mask 只用于评分，不参与选窗。

## 聚合（10 张篡改图）

- A：mean coverage 0.7866，mean window area 0.4751，mean window mask frac 0.6697。
- B：mean coverage 0.4637，mean window area 0.2452，mean window mask frac 0.7242。
- C：mean union coverage 0.4675，mean union area 0.2463，mean window mask frac 0.7413。
- B coverage ≥ A 的 95% 的图：5/10（41993, 42958, 57143, 60788, 67065）。
- B 窗口面积小于 A 的图：6/10（40489, 43686, 56086, 57276, 60788, 73275）；这些图里能保住 A 的 95% 覆盖的只有 60788。
- B 与 A 完全同一框（高响应只有一个主导连通分量）：41993, 42958, 57143, 67065。
- C coverage ≥ A 的 95% 的图：5/10（41993, 42958, 57143, 60788, 67065）。
- C 第二框新增覆盖 > 0 的图：5/10（40489, 43686, 56086, 60788, 73275）。

## 逐图（篡改）

| sample | A cov / area | B cov / area | C union cov / area | C second +cov / +area |
|---|---:|---:|---:|---:|
| `40489` | 0.377611 / 0.258705 | 0.19241 / 0.112087 | 0.20024 / 0.116451 | 0.00783 / 0.004364 |
| `41993` | 0.965475 / 0.780341 | 0.965475 / 0.780341 | 0.965475 / 0.780341 | None / None |
| `42958` | 1.0 / 0.188324 | 1.0 / 0.188324 | 1.0 / 0.188324 | None / None |
| `43686` | 1.0 / 0.962858 | 0.003804 / 0.00208 | 0.005097 / 0.002787 | 0.001293 / 0.000707 |
| `56086` | 0.768378 / 0.166097 | 0.003935 / 0.000916 | 0.030035 / 0.004985 | 0.0261 / 0.004069 |
| `57143` | 0.981419 / 0.29935 | 0.981419 / 0.29935 | 0.981419 / 0.29935 | None / None |
| `57276` | 0.305453 / 0.187135 | 0.001581 / 0.001519 | 0.001581 / 0.001519 | None / None |
| `60788` | 0.501 / 0.428602 | 0.48661 / 0.417752 | 0.487021 / 0.418062 | 0.000411 / 0.00031 |
| `67065` | 1.0 / 0.648499 | 1.0 / 0.648499 | 1.0 / 0.648499 | 0.0 / 0.0 |
| `73275` | 0.966645 / 0.830841 | 0.001488 / 0.000916 | 0.004264 / 0.002909 | 0.002776 / 0.001993 |

## 真图窗口

| sample | A window / context | B window / context | C window / context |
|---|---|---|---|
| `42307` | area 0.205195, border=True, context=border, edge×0.7194, texture×0.1402 | area 0.005296, border=False, context=content, edge×1.335, texture×0.223 | area 0.202597, border=True, context=border, edge×0.7181, texture×0.1413 |
| `47630` | area 0.357229, border=True, context=border, edge×1.2842, texture×0.3083 | area 0.0034, border=True, context=border+edge, edge×1.8384, texture×0.229 | area 0.004095, border=True, context=border+edge, edge×1.8751, texture×0.2468 |
| `52191` | area 0.03599, border=False, context=edge, edge×1.5285, texture×0.495 | area 0.03599, border=False, context=edge, edge×1.5285, texture×0.495 | area 0.03599, border=False, context=edge, edge×1.5285, texture×0.495 |
| `65749` | area 0.174863, border=True, context=border, edge×0.9768, texture×0.6057 | area 0.00651, border=True, context=border, edge×1.3047, texture×0.2989 | area 0.00651, border=True, context=border, edge×1.3047, texture×0.2989 |
| `70102` | area 0.044491, border=True, context=border, edge×0.0, texture×0.0188 | area 0.006397, border=True, context=border, edge×0.0, texture×0.1034 | area 0.014423, border=True, context=border, edge×0.0, texture×0.0524 |

## Top-2 新增框

| sample | 第二框 box | 第二框面积占比 | +coverage | +area |
|---|---|---:|---:|---:|
| `40489` | (182, 110, 205, 127) | 0.004364 | 0.00783 | 0.004364 |
| `43686` | (341, 357, 351, 367) | 0.000707 | 0.001293 | 0.000707 |
| `56086` | (158, 266, 174, 291) | 0.004069 | 0.0261 | 0.004069 |
| `60788` | (218, 109, 233, 124) | 0.002325 | 0.000411 | 0.00031 |
| `73275` | (370, 80, 384, 94) | 0.001994 | 0.002776 | 0.001993 |

Top-2 只在这 5 张图有非零新增；最大新增是 56086 的 +2.61 个百分点覆盖。其余图只有一个有效连通分量（或第二分量完全落在第一框内），所以 C=A/B。

## 代表图

`representative/` 下为原图 + 异常图，A 蓝、B/C Top-1 绿、C Top-2 橙；不显示 mask。

## 结论

- **Top-1 不能可靠地用更小窗口替代 A。** 10 张里 5 张 B 的 coverage 达到 A 的 95%，但其中 4 张 B 与 A 是同一个框（41993/42958/57143/67065）；只有 60788 略小（面积 0.418 vs 0.429，coverage 0.487 vs 0.501），收益很小。其余 5 张 B 虽然面积更小，coverage 却塌到 0.0016–0.192（A 为 0.305–1.000）。
- **Top-2 补不回 A 的覆盖。** 只有 5 张有非零新增，最大是 56086 的+2.61 个百分点；C 的并集平均 coverage 0.4675，仍远低于 A 的 0.7866。43686/73275 的高响应碎成 9/7 个连通分量，Top-2 仍只覆盖很小一角。
- **真图窗口主要落在边框/边缘，不在高纹理上。** A 在 42307/47630/65749/70102触边，52191 在高边缘密度处；5 张真图的 A texture× 都 <1（0.02–0.61）。B/C 的真图窗口多数更小且继续贴边（47630 的 edge× 达 1.84/1.88），没有变成纹理窗口。
- 因此当前 A（所有 high-response 像素的外接框）在这批图上仍是覆盖最稳的方案；B/C 可以作为更小的辅助定位，但不适合替代三图主裁剪。

## 边界

- 只复用已有的 TruFor map/conf 和 15 张验证图；未重跑 Qwen/Agent/TruFor。
- 真图没有 mask，覆盖率列为空；只报告窗口是否触边/高纹理（相对整图）。
- Top-1/Top-2 的排序分数是透明的 `mean anomaly × mean confidence`；不是新的专家分数。
