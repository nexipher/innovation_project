# Dual-expert control vs new batch — descriptive comparison

> This is **not** a human acceptance verdict. Heuristic matches are listed
> below; box/text mismatch is a candidate list and must be checked against
> the overlay images.  Negated phrases ("无需交叉验证") and visual/map
> agreement ("与观察高度吻合") are separated from cross-expert claims;
> a complaint that a map lacks coordinates is not a missing-image claim.

| 指标 | 旧 24 条（控制） | 新批次 | Δ（新−旧） |
| --- | ---: | ---: | ---: |
| sessions | 24 | 24 | +0 |
| 整图判断完全正确 | 16 | 11 | -5 |
| 局部判断完全正确 | 6 | 7 | +1 |
| 两维都正确 | 2 | 3 | +1 |
| Real/Not-tampered 被误判 Tampered | 2 | 3 | +1 |
| joint_verdict_issues 会话 | 0 | 0 | +0 |
| 局部结论缺 TruFor 测量 | 0 | 0 | +0 |
| expert call 数为 2 的会话 | 23 | 9 | -14 |
| TruFor 先调用的会话 | 0 | 16 | +16 |
| 图片误读候选（有图却称未提供/看不到） | 1 | 0 | -1 |
| 抱怨 map 没给坐标/可靠性（不是未投送） | 4 | 0 | -4 |
| 强跨任务相互验证措辞（排除否定句） | 1 | 1 | +0 |
| 其中被否定的相互验证句（只列不计） | 0 | 2 | +2 |
| 视觉观察与 map 的相似措辞（非跨专家） | 0 | 1 | +1 |
| 弱‘方向/结果一致’措辞 | 9 | 3 | -6 |
| 总体 Uncertain | 4 | 10 | +6 |
| 笼统 Uncertain（无具体限制） | 0 | 0 | +0 |
| 框文不一致候选 | 1 | 0 | -1 |
| 文字提到异常区/高响应但无框 | 4 | 5 | +1 |
| 可疑区域数 | 10 | 10 | +0 |
| 全图 bbox 数 | 3 | 3 | +0 |
| confidence=0.95 会话 | 17 | 10 | -7 |
| confidence>=0.9 会话 | 18 | 10 | -8 |
| confidence mean | 0.9096 | 0.8104 | -0.0992 |
| confidence stdev | 0.0768 | 0.1683 | +0.0915 |

## 候选明细

### 新批次：图片误读（明确称图未提供）
无自动命中。

### 新批次：map 缺坐标/可靠性抱怨（不是未投送）
无自动命中。

### 新批次：强相互验证（排除否定句）
- `autosplice__39890__1__q95`: 相互支持

### 新批次：被否定的相互验证句（不计为违规）
- `genimage__SD14__045_sdv4_00039__q95`: 交叉验证
- `autosplice__72399__2__q95`: 互相验证

### 新批次：视觉观察与 map 的相似措辞（非跨专家）
- `genimage__ADM__146_adm_91__q95`: 高度吻合

### 新批次：弱一致措辞
- `autosplice__58610__authentic__q95`: consistent
- `autosplice__42165__authentic__q95`: 结果一致
- `autosplice__57501__0__q95`: 无冲突

### 新批次：笼统 Uncertain
无自动命中。

### 新批次：框文不一致
无自动命中。

### 新批次：文字提到异常区/高响应但无框
- `autosplice__51136__authentic__q95`: 致。因此，无法通过现有证据明确区分是AI生成还是局部编辑，且PROBE的高分可能误判。结论应为“Uncertain”，因证据存在冲突且无法排除模型误判或局部编辑的解释。
</reasoning>
<verdict>
{"verdict": "Uncertain", "whole_image_generation": "Uncertain", "local_m
- `autosplice__53059__authentic__q95`: 由于 TruFor 的局部检测结果不可靠，且 PROBE 的高分是图像级判断，当前证据不足以支持“局部篡改”的明确结论，但足以支持“AI 生成”的判断。因此，最终结论为图像整体为 AI 生成，局部篡改证据不足，结论为“Uncertain”。
</reasoning>
<verdict>
{"verdict": "Uncertain", "whole_imag
- `genimage__BigGAN__051_biggan_00020__q95`: 但 TruFor 的输出明确指出，该分数和图像是局部编辑的候选，而非全图 AI 生成的判断。当前无视觉证据显示图像存在明显的局部编辑或 AI 生成痕迹，且 TruFor 的低分支持图像未被显著篡改。由于尚未调用 PROBE，无法判断图像整体是否为 AI 生成。当前证据不足以支持“局部编辑”或“AI 生成”的结论，但亦无明确证据支持“未被篡改”或“真实”。
<
- `genimage__Midjourney__144_midjourney_34__q95`: 断图像整体是否为 AI 生成。当前图像视觉上自然，无明显AI生成痕迹，但 PROBE 未被调用，因此无法确认整体生成性。由于缺少 TruFor 的测量，无法判断局部操纵情况，且 PROBE 未执行，无法判断整体生成性。因此，结论必须为“Uncertain”，并明确说明缺失 TruFor 测量限制了局部操纵判断，缺少 PROBE 测量限制了整体生成性判断。
<
- `autosplice__42005__0__q95`: 生成判断。当前证据仅来自 TruFor，未进行跨专家比较，且 PROBE 未调用，因此无法完成对“whole_image_generation”和“local_manipulation”的完整判断。
</reasoning>
<verdict>
{"verdict": "Uncertain", "whole_image_generation": "Uncer

### 全图 bbox 明细（人工核对这些框是否被写成局部定位）
- `genimage__ADM__146_adm_91__q95` scope=global_range: 鸟下方黑色物体形态异常，与飞行姿态和物理逻辑不符，TruFor 检测到高局部篡改响应（0.9858）且置信度高，表明该区域为后期添加或合成。
- `genimage__SD14__045_sdv4_00039__q95` scope=global_range: TruFor异常图显示主体区域为高响应，置信度高，表明存在局部编辑嫌疑；PROBE高分指向整图AI生成，二者不冲突。
- `autosplice__42005__2__q95` scope=global_range: TruFor 的局部篡改检测分数高（0.94143），且异常图在整图区域呈现高响应（红色），但其高分本身不构成“已确认篡改”，仅是“候选”；PROBE 的 AI 生成分数高（0.9857），但 TruFor 的高响应区域覆盖整图，可能与 AI 生成的全局特征相关，无法确认是否为“局部篡改”或“AI 生成”中的“局部编辑”

