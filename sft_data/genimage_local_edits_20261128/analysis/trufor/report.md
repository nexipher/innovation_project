# 10 对 GenImage 整图生成/局部编辑 — 独立 TruFor

阈值：score >= 0.5 记图像级检出；IoU >= 0.1 记定位命中；mask 面积占比 >= 0.3 记大范围编辑（p05 单列）。原图误报指未编辑原图的 score 或同 mask 区域响应被判为局部篡改。

## 聚合

| 指标 | 全部 10 对 | 不含 p05（9 对） |
|---|---:|---:|
| 编辑版图像级检出（score>=0.5） | 0/10 | 0/9 |
| 编辑版定位命中（IoU>=0.1） | 2/10 | 2/9 |
| 编辑版 mask 高响应占比>=0.1 | 2/10 | 2/9 |
| 编辑版检出且定位 | 0/10 | 0/9 |
| 编辑版 mask 均值高于原图 | 7/10 | — |
| 原图 score 误报 | 0/10 | 0/9 |
| 原图同区域定位误报 | 0/10 | 0/9 |

## 逐对

| pair | 类型 | 大编辑 | 原 score | 编 score | Δscore | 原 mask mean | 编 mask mean | 编 IoU | 编检出/定位 | 原 FP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `p01_vqdm_box_label_add` | add |  | 0.317 | 0.095 | -0.222 | 0.002 | 0.205 | 0.124 | N/Y | N |
| `p02_wukong_tag_add` | add |  | 0.060 | 0.206 | +0.146 | 0.163 | 0.569 | 0.602 | N/Y | N |
| `p03_sd14_sphere_add` | add |  | 0.436 | 0.446 | +0.011 | 0.000 | 0.133 | 0.036 | N/N | N |
| `p04_glide_556_text_add` | add |  | 0.150 | 0.148 | -0.002 | 0.116 | 0.239 | 0.000 | N/N | N |
| `p05_biggan_stick_delete` | delete | Y | 0.163 | 0.307 | +0.144 | 0.017 | 0.086 | 0.000 | N/N | N |
| `p06_glide_922_paragraph_delete` | delete |  | 0.010 | 0.012 | +0.003 | 0.002 | 0.002 | 0.000 | N/N | N |
| `p07_glide_830_caption_delete` | delete |  | 0.184 | 0.317 | +0.133 | 0.103 | 0.041 | 0.000 | N/N | N |
| `p08_adm_notice_replace` | replace |  | 0.166 | 0.215 | +0.050 | 0.146 | 0.091 | 0.000 | N/N | N |
| `p09_glide_screens_replace` | replace |  | 0.080 | 0.128 | +0.048 | 0.117 | 0.134 | 0.043 | N/N | N |
| `p10_vqdm_document_replace` | replace |  | 0.120 | 0.122 | +0.003 | 0.019 | 0.046 | 0.000 | N/N | N |

## p05 单列

p05 编辑 mask 覆盖 45.5% 的图像面积；原图 score 0.163，编辑版 score 0.307，mask mean 0.017→0.086，IoU 0.000。大范围编辑的分数/响应不能与小块编辑直接平均。

## 产物

- `per_pair.csv`：逐对 score、mask 响应、IoU、检出/误报标志。
- `trufor_analysis.json`：完整指标与聚合。
- `trufor_maps/*.png`：每张图的 anomaly/confidence 图和同 mask 区域三图。
- 本轮只跑独立 TruFor；Agent 结果在 `analysis/agent/` 另行汇总。
