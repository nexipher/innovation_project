# EditBench 5 对 — base Agent vs 独立 TruFor

同一批 10 张 JPEG-q95；Agent 沿用当前 PROBE+TruFor、裁剪和自主调用，未改规则；mask 仅用于离线评估。

## 聚合

| 指标 | 结果 |
|---|---:|
| 整图 AI-generated 判对 | 7/10 |
| 编辑图整图 AI-generated | 3/5 |
| 编辑图局部 Tampered | 1/5 |
| 编辑图 both（整图 AI + 局部 Tampered） | 0/5 |
| 未编辑原图局部误报（Tampered） | 1/5 |
| 未编辑原图局部 Uncertain | 1/5 |
| 总体弃权（Uncertain） | 0/10 |
| 编辑图总体弃权 | 0/5 |
| 原图总体弃权 | 0/5 |
| TruFor 调用 | 10/10 |
| 专家调用总数 / 平均每图 | 19 / 1.9 |

整图错误 3/10：`eb_gen_015_sawfish__edited` whole=Uncertain（只调用了 TruFor，未调用 PROBE）；`eb_gen_114_three__original` 与 `eb_gen_114_three__edited` whole=Real（PROBE 0.0908 / 0.1482）。

## 逐图对照

| pair | variant | TF score | TF mask mean | TF IoU | Agent whole | Agent local | Agent overall | TF called |
|---|---|---:|---:|---:|---|---|---|---|
| `eb_gen_015_sawfish` | edited | 0.601 | 0.593 | 0.601 | Uncertain | Tampered | Fake | Y |
| `eb_gen_015_sawfish` | original | 0.947 | 0.000 | 0.000 | AI-generated | Tampered | Fake | Y |
| `eb_gen_025_dog` | edited | 0.473 | 0.016 | 0.000 | AI-generated | Not tampered | Fake | Y |
| `eb_gen_025_dog` | original | 0.275 | 0.031 | 0.001 | AI-generated | Not tampered | Fake | Y |
| `eb_gen_029_pig` | edited | 0.115 | 0.276 | 0.187 | AI-generated | Uncertain | Fake | Y |
| `eb_gen_029_pig` | original | 0.160 | 0.038 | 0.000 | AI-generated | Not tampered | Fake | Y |
| `eb_gen_064_triangle` | edited | 0.469 | 0.332 | 0.293 | AI-generated | Not tampered | Fake | Y |
| `eb_gen_064_triangle` | original | 0.152 | 0.153 | 0.070 | AI-generated | Uncertain | Fake | Y |
| `eb_gen_114_three` | edited | 0.336 | 0.515 | 0.420 | Real | Uncertain | Real | Y |
| `eb_gen_114_three` | original | 0.265 | 0.576 | 0.537 | Real | Not tampered | Real | Y |

## 重点案例

### eb_gen_029_pig / edited

- 真值：whole AI-generated，local Tampered。
- TruFor：score 0.115，mask mean 0.276，mask high 0.191，IoU 0.187；低分热图覆盖=True。
- Agent：whole **AI-generated** / local **Uncertain** / overall **Fake**；calls 2；TruFor called True (strength 0.1149, metric 0.114912)。
- 结案依据：主要支持：PROBE 高分（0.917）支持整图 AI 生成；主要限制：TruFor 分数低（0.115）且高响应区域可靠性低，无法确认局部编辑。
- 热图：`analysis/trufor/previews/eb_gen_029_pig.jpg`
- trace：`sft_data/editbench_verification_20261128/analysis/agent/agent/traces/forensic_sft_session_20261128_210249_eb_gen_029_pig__edited.json`

### eb_gen_064_triangle / edited

- 真值：whole AI-generated，local Tampered。
- TruFor：score 0.469，mask mean 0.332，mask high 0.293，IoU 0.293；低分热图覆盖=True。
- Agent：whole **AI-generated** / local **Not tampered** / overall **Fake**；calls 2；TruFor called True (strength 0.4686, metric 0.468612)。
- 结案依据：主要支持：PROBE 显示图像为 AI 生成（分数 0.9188），TruFor 未检测到局部篡改（分数 0.4686）。主要限制：局部编辑结论基于 TruFor 的低响应，但图像存在不自然感，可能暗示生成过程中的合成痕迹。
- 热图：`analysis/trufor/previews/eb_gen_064_triangle.jpg`
- trace：`sft_data/editbench_verification_20261128/analysis/agent/agent/traces/forensic_sft_session_20261128_210814_eb_gen_064_triangle__edited.json`

### eb_gen_015_sawfish / original

- 真值：whole AI-generated，local Not tampered。
- TruFor：score 0.947，mask mean 0.000，mask high 0.000，IoU 0.000；低分热图覆盖=False。
- Agent：whole **AI-generated** / local **Tampered** / overall **Fake**；calls 2；TruFor called True (strength 0.9466, metric 0.946601)。
- 结案依据：主要支持：PROBE高分（0.9511）指向AI生成，TruFor高分（0.9466）支持局部篡改；主要限制：TruFor高响应区域位于水印，非中央悬浮物体，但足以支持局部编辑。
- 热图：`analysis/trufor/previews/eb_gen_015_sawfish.jpg`
- trace：`sft_data/editbench_verification_20261128/analysis/agent/agent/traces/forensic_sft_session_20261128_210533_eb_gen_015_sawfish__original.json`

## 产物

- `per_image.csv`：10 行，逐图 TruFor 与 Agent 对照。
- `comparison.json`：完整聚合和 Agent 记录。
- 判定依据中的 Agent 报告直接来自 `analysis/agent/agent/records.json`；热图见 `analysis/trufor/previews/`。

## 边界

本轮只跑 base Agent；未改规则、未训练、未安装 PIM。TruFor 的 0.947/0.601 等分数来自上一轮独立 JPEG-q95 运行。
