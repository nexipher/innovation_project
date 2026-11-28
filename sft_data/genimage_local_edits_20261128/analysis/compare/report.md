# 10 对 GenImage 整图生成/局部编辑 — TruFor + Agent

固定当前 PROBE + TruFor、high-response 裁剪和自主调用流程；未改规则、未训练。
TruFor 先独立跑 20 张，再用同一批图跑 Agent（原版 Qwen3-VL）。

## 独立 TruFor

| 指标 | 全部 10 对 | 不含 p05（9 对） |
|---|---:|---:|
| 编辑版图像级检出（score>=0.5） | 0/10 | 0/9 |
| 编辑版定位命中（IoU>=0.1） | 2/10 | 2/9 |
| 编辑版 mask 高响应占比>=0.1 | 2/10 | 2/9 |
| 编辑版检出且定位 | 0/10 | 0/9 |
| 原图 score 误报 | 0/10 | 0/9 |
| 原图同区域定位误报 | 0/10 | 0/9 |

## Agent

| 指标 | 全部 10 对 | 不含 p05（9 对） |
|---|---:|---:|
| 编辑版整图 AI-generated | 8/10 | 8/9 |
| 原图整图 AI-generated | 9/10 | 8/9 |
| 20 张整图 AI-generated | 17/20 | 16/18 |
| 编辑版局部 Tampered | 0/10 | 0/9 |
| 编辑版同时判对（whole AI + local Tampered） | 0/10 | 0/9 |
| 编辑版局部 Uncertain | 0/10 | 0/9 |
| 原图局部误报（Tampered） | 0/10 | 0/9 |
| 原图局部 Uncertain | 2/10 | 2/9 |
| 总体弃权（20 张） | 0/20 | 0/18 |
| 编辑版调用了 TruFor | 10/10 | 9/9 |
| 专家调用总数 / 平均每图 | 40 / 2.0 | 36 / 2.0 |

整图 AI 错误 3/20：p05 编辑版（PROBE 0.490，整图 Real）、p07 原图与编辑版（PROBE 0.25，整图 Real）。

分类：`detected`（局部 Tampered）、`trufor_not_localized`（独立 TruFor 未达到可用证据）、`agent_not_called`（TruFor 可用但 Agent 未调用）、`usable_but_missed`（TruFor 可用且 Agent 调用但仍未写 Tampered）、`detected_without_trufor_evidence`（独立 TruFor 未命中但 Agent 仍写 Tampered）。

## 逐对

| pair | 类型 | TF orig→edit score | TF mask mean | TF IoU | Agent 原图 局部/总体 | Agent 编辑版 整图/局部/总体 | both | class |
|---|---|---|---:|---:|---|---|---|---|
| `p01_vqdm_box_label_add` | add | 0.317→0.095 | 0.002→0.205 | 0.124 | Uncertain/Fake | AI-generated/Not tampered/Fake | N | usable_but_missed |
| `p02_wukong_tag_add` | add | 0.060→0.206 | 0.163→0.569 | 0.602 | Not tampered/Fake | AI-generated/Not tampered/Fake | N | usable_but_missed |
| `p03_sd14_sphere_add` | add | 0.436→0.446 | 0.000→0.133 | 0.036 | Uncertain/Fake | AI-generated/Not tampered/Fake | N | trufor_not_localized |
| `p04_glide_556_text_add` | add | 0.150→0.148 | 0.116→0.239 | 0.000 | Not tampered/Fake | AI-generated/Not tampered/Fake | N | trufor_not_localized |
| `p05_biggan_stick_delete` | delete | 0.163→0.307 | 0.017→0.086 | 0.000 | Not tampered/Fake | Real/Not tampered/Real | N | trufor_not_localized |
| `p06_glide_922_paragraph_delete` | delete | 0.010→0.012 | 0.002→0.002 | 0.000 | Not tampered/Fake | AI-generated/Not tampered/Fake | N | trufor_not_localized |
| `p07_glide_830_caption_delete` | delete | 0.184→0.317 | 0.103→0.041 | 0.000 | Not tampered/Real | Real/Not tampered/Real | N | trufor_not_localized |
| `p08_adm_notice_replace` | replace | 0.166→0.215 | 0.146→0.091 | 0.000 | Not tampered/Fake | AI-generated/Not tampered/Fake | N | trufor_not_localized |
| `p09_glide_screens_replace` | replace | 0.080→0.128 | 0.117→0.134 | 0.043 | Not tampered/Fake | AI-generated/Not tampered/Fake | N | trufor_not_localized |
| `p10_vqdm_document_replace` | replace | 0.120→0.122 | 0.019→0.046 | 0.000 | Not tampered/Fake | AI-generated/Not tampered/Fake | N | trufor_not_localized |

## 错因分桶

- **detected**（0）：—
- **detected_without_trufor_evidence**（0）：—
- **usable_but_missed**（2）：`p01_vqdm_box_label_add`, `p02_wukong_tag_add`
- **agent_not_called**（0）：—
- **trufor_not_localized**（8）：`p03_sd14_sphere_add`, `p04_glide_556_text_add`, `p05_biggan_stick_delete`, `p06_glide_922_paragraph_delete`, `p07_glide_830_caption_delete`, `p08_adm_notice_replace`, `p09_glide_screens_replace`, `p10_vqdm_document_replace`

原图局部误报：无
编辑版未调用 TruFor：无

## 代表案例

见 `representatives/index.html`：`p02_wukong_tag_add`, `p01_vqdm_box_label_add`, `p05_biggan_stick_delete`, `p07_glide_830_caption_delete`, `p03_sd14_sphere_add`

## p05 单列
p05 mask 覆盖 45.5% 图像面积；TruFor 原/编 score 0.163→0.307，mask mean 0.017→0.086，IoU 0.000；Agent 编辑版 Real/Not tampered/Real。大范围编辑未形成高响应，且 PROBE 在编辑后把整图判成 Real；不能与小块编辑平均。

## 边界

本轮只跑独立 TruFor 和当前 base Agent；未改规则、未训练、未扩样。p05 mask 覆盖 45.5% 图像面积，已在大表和小表中单列；其分数/响应不与小块编辑直接平均。
