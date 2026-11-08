# Dual-expert control vs new batch — descriptive comparison

> This is **not** a human acceptance verdict. Heuristic matches are listed
> below; box/text mismatch is a candidate list and must be checked against
> the overlay images.  Negated phrases ("无需交叉验证") and visual/map
> agreement ("与观察高度吻合") are separated from cross-expert claims;
> a complaint that a map lacks coordinates is not a missing-image claim.

| 指标 | 旧 24 条（控制） | 新批次 | Δ（新−旧） |
| --- | ---: | ---: | ---: |
| sessions | 24 | 24 | +0 |
| 整图判断完全正确 | 16 | 16 | +0 |
| 局部判断完全正确 | 11 | 11 | +0 |
| 两维都正确 | 8 | 8 | +0 |
| Real/Not-tampered 被误判 Tampered | 5 | 2 | -3 |
| joint_verdict_issues 会话 | 2 | 0 | -2 |
| 局部结论缺 TruFor 测量 | 0 | 0 | +0 |
| 仍有未尝试专家 | 1 | 2 | +1 |
| 有执行失败专家 | 1 | 1 | +0 |
| 触发结案前检查的会话 | 3 | 6 | +3 |
| 结案前检查次数 | 3 | 6 | +3 |
| 弃权仍引用未尝试专家 | 0 | 0 | +0 |
| 有框核验标记的会话 | 11 | 1 | -10 |
| 覆盖检查后重答的会话 | 3 | 4 | +1 |
| 覆盖检查失败/未纠正的会话 | 2 | 0 | -2 |
| expert call 数为 2 的会话 | 22 | 21 | -1 |
| TruFor 先调用的会话 | 20 | 20 | +0 |
| 图片误读候选（有图却称未提供/看不到） | 0 | 0 | +0 |
| 抱怨 map 没给坐标/可靠性（不是未投送） | 0 | 0 | +0 |
| 强跨任务相互验证措辞（排除否定句） | 1 | 1 | +0 |
| 其中被否定的相互验证句（只列不计） | 3 | 5 | +2 |
| 视觉观察与 map 的相似措辞（非跨专家） | 0 | 0 | +0 |
| 弱‘方向/结果一致’措辞 | 6 | 4 | -2 |
| 总体 Uncertain | 1 | 3 | +2 |
| 笼统 Uncertain（无具体限制） | 0 | 0 | +0 |
| 框文不一致候选 | 0 | 1 | +1 |
| 文字提到异常区/高响应但无框 | 2 | 3 | +1 |
| 可疑区域数 | 11 | 8 | -3 |
| 全图 bbox 数 | 11 | 0 | -11 |
| confidence=0.95 会话 | 18 | 17 | -1 |
| confidence>=0.9 会话 | 21 | 20 | -1 |
| confidence mean | 0.9412 | 0.9246 | -0.0166 |
| confidence stdev | 0.0359 | 0.0756 | +0.0397 |

## 候选明细

### 新批次：图片误读（明确称图未提供）
无自动命中。

### 新批次：map 缺坐标/可靠性抱怨（不是未投送）
无自动命中。

### 新批次：强相互验证（排除否定句）
- `genimage__ADM__146_adm_91__q95`: 相互支持

### 新批次：被否定的相互验证句（不计为违规）
- `genimage__Real__333c2446-4c1a-4c9f-b296-9c5b3c5fdf94__q95`: 交叉验证
- `genimage__Real__35b3b70e-4994-4618-8aaa-86467988d1d5__q95`: 交叉验证
- `autosplice__51136__authentic__q95`: 相互验证
- `autosplice__41187__0__q95`: 相互支持
- `autosplice__42005__2__q95`: 互相验证

### 新批次：视觉观察与 map 的相似措辞（非跨专家）
无自动命中。

### 新批次：弱一致措辞
- `genimage__Real__351a8be1-104f-49be-9634-7c045dee48a2__q95`: 结果一致
- `autosplice__58610__authentic__q95`: 无冲突
- `autosplice__42165__authentic__q95`: 结果一致
- `autosplice__57501__0__q95`: 无冲突

### 新批次：笼统 Uncertain
无自动命中。

### 新批次：框文不一致
- `genimage__ADM__146_adm_91__q95`: 框中心在 [645.0, 348.0]，文字却提到 天空

### 新批次：文字提到异常区/高响应但无框
- `autosplice__51136__authentic__q95`: 持图像整体为 AI 生成，而 TruFor 的高分支持局部存在编辑痕迹。两者结果一致，但不构成相互验证。由于两个专家均已返回有效测量，且结果一致，无需再调用其他工具。结论基于两个独立专家的测量结果，且无冲突。
</reasoning>
<verdict>
{"verdict": "Fake", "whole_image_generation": "AI-ge
- `genimage__SD14__045_sdv4_00039__q95`: ain”，因为 TruFor 的高分仅指向局部区域，且未提供编辑过程证据，同时其高响应区域的图像内容无明显人为编辑痕迹，因此无法确认是否为局部编辑。PROBE 的高分支持整图 AI 生成，但 TruFor 的局部高分不支持或否定局部编辑，因此局部编辑状态无法确定。
</reasoning>
<verdict>
{"verdict": "Fake", "who
- `genimage__Wukong__147_wukong_image128__q95`: ），局部异常可能源于生成过程中的细节处理，而非后期编辑。因此，PROBE的高分更可能反映整图生成，而TruFor的局部响应可能为生成过程中的自然变异，不足以支持“局部篡改”结论。最终结论需综合判断：整图生成可能性高，局部编辑证据不足。
</reasoning>
<verdict>
{"verdict": "Fake", "whole_image_genera

### 全图 bbox 明细（人工核对这些框是否被写成局部定位）
无全图 bbox。

