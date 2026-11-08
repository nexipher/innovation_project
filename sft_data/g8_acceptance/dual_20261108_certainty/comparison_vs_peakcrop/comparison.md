# Dual-expert control vs new batch — descriptive comparison

> This is **not** a human acceptance verdict. Heuristic matches are listed
> below; box/text mismatch is a candidate list and must be checked against
> the overlay images.  Negated phrases ("无需交叉验证") and visual/map
> agreement ("与观察高度吻合") are separated from cross-expert claims;
> a complaint that a map lacks coordinates is not a missing-image claim.

| 指标 | 旧 24 条（控制） | 新批次 | Δ（新−旧） |
| --- | ---: | ---: | ---: |
| sessions | 24 | 24 | +0 |
| 整图判断完全正确 | 16 | 14 | -2 |
| 局部判断完全正确 | 11 | 8 | -3 |
| 两维都正确 | 8 | 5 | -3 |
| Real/Not-tampered 被误判 Tampered | 2 | 4 | +2 |
| joint_verdict_issues 会话 | 0 | 7 | +7 |
| 局部结论缺 TruFor 测量 | 0 | 0 | +0 |
| 仍有未尝试专家 | 2 | 4 | +2 |
| 有执行失败专家 | 1 | 5 | +4 |
| 触发结案前检查的会话 | 6 | 11 | +5 |
| 结案前检查次数 | 6 | 11 | +5 |
| 弃权仍引用未尝试专家 | 0 | 0 | +0 |
| 有框核验标记的会话 | 1 | 0 | -1 |
| 覆盖检查后重答的会话 | 4 | 4 | +0 |
| 覆盖检查失败/未纠正的会话 | 0 | 1 | +1 |
| expert call 数为 2 的会话 | 21 | 15 | -6 |
| TruFor 先调用的会话 | 20 | 17 | -3 |
| 图片误读候选（有图却称未提供/看不到） | 0 | 1 | +1 |
| 抱怨 map 没给坐标/可靠性（不是未投送） | 0 | 0 | +0 |
| 强跨任务相互验证措辞（排除否定句） | 1 | 1 | +0 |
| 其中被否定的相互验证句（只列不计） | 5 | 0 | -5 |
| 视觉观察与 map 的相似措辞（非跨专家） | 0 | 0 | +0 |
| 弱‘方向/结果一致’措辞 | 4 | 7 | +3 |
| 总体 Uncertain | 3 | 7 | +4 |
| 笼统 Uncertain（无具体限制） | 0 | 0 | +0 |
| 框文不一致候选 | 1 | 0 | -1 |
| 文字提到异常区/高响应但无框 | 3 | 2 | -1 |
| 可疑区域数 | 8 | 7 | -1 |
| 全图 bbox 数 | 0 | 0 | +0 |
| certainty=High | 未记录 | 14 | — |
| certainty=Medium | 未记录 | 3 | — |
| certainty=Low | 未记录 | 0 | — |
| certainty=null（Uncertain） | 未记录 | 7 | — |
| 旧 confidence=0.95 会话 | 17 | 未记录 | — |
| 旧 confidence>=0.9 会话 | 20 | 未记录 | — |
| 旧 confidence mean | 0.9246 | 未记录 | — |
| 旧 confidence stdev | 0.0756 | 未记录 | — |

## 候选明细

### 新批次：图片误读（明确称图未提供）
- `autosplice__53059__authentic__q95`: 未达高阈值），其异常图显示面部区域有中等响应，可靠性图

### 新批次：map 缺坐标/可靠性抱怨（不是未投送）
无自动命中。

### 新批次：强相互验证（排除否定句）
- `autosplice__42005__2__q95`: 共同支持

### 新批次：被否定的相互验证句（不计为违规）
无自动命中。

### 新批次：视觉观察与 map 的相似措辞（非跨专家）
无自动命中。

### 新批次：弱一致措辞
- `genimage__Real__333c2446-4c1a-4c9f-b296-9c5b3c5fdf94__q95`: 无冲突
- `genimage__Real__339e8fae-3317-4d7f-9c6e-c36754bf35a7__q95`: 无冲突
- `genimage__Real__35b3b70e-4994-4618-8aaa-86467988d1d5__q95`: 无冲突
- `autosplice__58610__authentic__q95`: consistent
- `autosplice__51136__authentic__q95`: consistent
- `autosplice__42165__authentic__q95`: 结果一致
- `autosplice__41187__0__q95`: consistent

### 新批次：笼统 Uncertain
无自动命中。

### 新批次：框文不一致
无自动命中。

### 新批次：文字提到异常区/高响应但无框
- `autosplice__72399__2__q95`: 对整图生成的判断。因此，当前证据支持“Fake”（因整图生成被判定为 AI 生成），但局部篡改证据为“候选”而非确证，且未见明确编辑痕迹。由于 PROBE 的高分与 TruFor 的“候选”结果在证据强度上不冲突，且 PROBE 的测量结果直接支持“Fake”，而 TruFor 的结果未改变“Fake”的整体判断，因此最终结论为“Fake”，主要支持来自 P
- `autosplice__57501__0__q95`: 体背景，可靠性高但无主体内容异常，无法确认是否存在局部篡改。由于 PROBE 的低分支持整图真实，而 TruFor 的高分无法确认局部篡改，两者证据方向不一致，但 PROBE 的低分更直接支持整图真实，且 TruFor 的高分区域非主体，其异常响应可能源于背景或压缩，而非主体篡改。因此，整图生成判断为 Real，局部编辑判断为 Uncertain，但整体结论

### 全图 bbox 明细（人工核对这些框是否被写成局部定位）
无全图 bbox。

