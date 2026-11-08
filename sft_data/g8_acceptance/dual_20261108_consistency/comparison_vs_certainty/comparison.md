# Dual-expert control vs new batch — descriptive comparison

> This is **not** a human acceptance verdict. Heuristic matches are listed
> below; box/text mismatch is a candidate list and must be checked against
> the overlay images.  Negated phrases ("无需交叉验证") and visual/map
> agreement ("与观察高度吻合") are separated from cross-expert claims;
> a complaint that a map lacks coordinates is not a missing-image claim.

| 指标 | 旧 24 条（控制） | 新批次 | Δ（新−旧） |
| --- | ---: | ---: | ---: |
| sessions | 24 | 24 | +0 |
| 整图判断完全正确 | 14 | 14 | +0 |
| 局部判断完全正确 | 8 | 7 | -1 |
| 两维都正确 | 5 | 5 | +0 |
| Real/Not-tampered 被误判 Tampered | 4 | 1 | -3 |
| joint_verdict_issues 会话 | 7 | 0 | -7 |
| 局部结论缺 TruFor 测量 | 0 | 0 | +0 |
| 仍有未尝试专家 | 4 | 10 | +6 |
| 有执行失败专家 | 5 | 2 | -3 |
| 触发结案前检查的会话 | 11 | 7 | -4 |
| 结案前检查次数 | 11 | 7 | -4 |
| 弃权仍引用未尝试专家 | 0 | 0 | +0 |
| 有框核验标记的会话 | 0 | 0 | +0 |
| 覆盖检查后重答的会话 | 4 | 3 | -1 |
| 覆盖检查失败/未纠正的会话 | 1 | 0 | -1 |
| certainty 字段检查通过 | 未记录 | 23 | — |
| certainty 字段检查重答后通过 | 未记录 | 1 | — |
| certainty 字段检查失败/未纠正 | 未记录 | 0 | — |
| expert call 数为 2 的会话 | 15 | 12 | -3 |
| TruFor 先调用的会话 | 17 | 14 | -3 |
| 图片误读候选（有图却称未提供/看不到） | 1 | 0 | -1 |
| 抱怨 map 没给坐标/可靠性（不是未投送） | 0 | 0 | +0 |
| 强跨任务相互验证措辞（排除否定句） | 1 | 0 | -1 |
| 其中被否定的相互验证句（只列不计） | 0 | 0 | +0 |
| 视觉观察与 map 的相似措辞（非跨专家） | 0 | 0 | +0 |
| 弱‘方向/结果一致’措辞 | 7 | 7 | +0 |
| 总体 Uncertain | 7 | 1 | -6 |
| 笼统 Uncertain（无具体限制） | 0 | 1 | +1 |
| 框文不一致候选 | 0 | 0 | +0 |
| 文字提到异常区/高响应但无框 | 2 | 3 | +1 |
| 可疑区域数 | 7 | 5 | -2 |
| 全图 bbox 数 | 0 | 0 | +0 |
| certainty=High | 14 | 21 | +7 |
| certainty=Medium | 3 | 2 | -1 |
| certainty=Low | 0 | 0 | +0 |
| certainty=null（Uncertain） | 7 | 1 | -6 |
| 旧 confidence=0.95 会话 | 未记录 | 未记录 | — |
| 旧 confidence>=0.9 会话 | 未记录 | 未记录 | — |
| 旧 confidence mean | 未记录 | 未记录 | — |
| 旧 confidence stdev | 未记录 | 未记录 | — |

## 候选明细

### 新批次：图片误读（明确称图未提供）
无自动命中。

### 新批次：map 缺坐标/可靠性抱怨（不是未投送）
无自动命中。

### 新批次：强相互验证（排除否定句）
无自动命中。

### 新批次：被否定的相互验证句（不计为违规）
无自动命中。

### 新批次：视觉观察与 map 的相似措辞（非跨专家）
无自动命中。

### 新批次：弱一致措辞
- `genimage__Real__333c2446-4c1a-4c9f-b296-9c5b3c5fdf94__q95`: 结果一致
- `genimage__Real__35b3b70e-4994-4618-8aaa-86467988d1d5__q95`: 结果一致
- `autosplice__42165__authentic__q95`: 无冲突
- `genimage__BigGAN__051_biggan_00020__q95`: 无冲突
- `autosplice__41187__0__q95`: consistent
- `autosplice__41187__1__q95`: consistent
- `autosplice__72399__2__q95`: 方向一致

### 新批次：笼统 Uncertain
- `autosplice__57501__0__q95`: PROBE 支持图像为真实拍摄，但 TruFor 的高响应区域与图像主体无关，且未提供具体编辑内容，因此局部篡改证据不足，整体结论为不确定。

### 新批次：框文不一致
无自动命中。

### 新批次：文字提到异常区/高响应但无框
- `autosplice__53059__authentic__q95`: 。然而，TruFor 的异常响应未明确指向具体编辑内容，仅表明存在局部编辑的可能，但未提供足够证据支持“Tampered”结论。因此，虽然PROBE的高分支持“Fake”，但TruFor的“Uncertain”局部编辑结论仍存在，且未明确编辑内容，导致整体结论仍为“Uncertain”。PROBE的高分是关键证据，但TruFor的“Uncertain”局部编
- `autosplice__42005__0__q95`: 返回 manipulation_anomaly_score=0.595108，高于阈值但其高响应区域可靠性低，且无具体编辑内容可确认，故支持“未发现局部篡改”。整图生成判断由 PROBE 支持为 AI 生成，局部编辑判断为“未发现篡改”，二者不矛盾：AI 生成图像可无局部篡改。因此，整体结论应为 Fake，因证据支持至少一个目标（整图生成）。
</reaso
- `autosplice__57501__0__q95`: 值 0.5，指向“Real”，而 TruFor 专家返回的局部篡改检测分数为 0.98934，但其高响应区域与图像主体无关，且未提供具体编辑内容，因此不能据此断定图像存在目标局部篡改。整体结论为“Uncertain”，因为局部篡改证据不足，且虽然 PROBE 支持真实，但 TruFor 的高响应区域与图像主体无关，且未提供具体编辑内容，因此不能排除局部篡改的

### 全图 bbox 明细（人工核对这些框是否被写成局部定位）
无全图 bbox。

