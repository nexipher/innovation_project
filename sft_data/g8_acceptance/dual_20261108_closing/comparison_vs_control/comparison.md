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
| 局部判断完全正确 | 6 | 12 | +6 |
| 两维都正确 | 2 | 7 | +5 |
| Real/Not-tampered 被误判 Tampered | 2 | 5 | +3 |
| joint_verdict_issues 会话 | 0 | 1 | +1 |
| 局部结论缺 TruFor 测量 | 0 | 1 | +1 |
| 仍有未尝试专家 | 未记录 | 5 | — |
| 有执行失败专家 | 未记录 | 1 | — |
| 触发结案前检查的会话 | 未记录 | 3 | — |
| 结案前检查次数 | 未记录 | 3 | — |
| 弃权仍引用未尝试专家 | 未记录 | 0 | — |
| expert call 数为 2 的会话 | 23 | 18 | -5 |
| TruFor 先调用的会话 | 0 | 10 | +10 |
| 图片误读候选（有图却称未提供/看不到） | 1 | 0 | -1 |
| 抱怨 map 没给坐标/可靠性（不是未投送） | 4 | 1 | -3 |
| 强跨任务相互验证措辞（排除否定句） | 1 | 3 | +2 |
| 其中被否定的相互验证句（只列不计） | 0 | 0 | +0 |
| 视觉观察与 map 的相似措辞（非跨专家） | 0 | 0 | +0 |
| 弱‘方向/结果一致’措辞 | 9 | 4 | -5 |
| 总体 Uncertain | 4 | 1 | -3 |
| 笼统 Uncertain（无具体限制） | 0 | 0 | +0 |
| 框文不一致候选 | 1 | 1 | +0 |
| 文字提到异常区/高响应但无框 | 4 | 4 | +0 |
| 可疑区域数 | 10 | 13 | +3 |
| 全图 bbox 数 | 3 | 3 | +0 |
| confidence=0.95 会话 | 17 | 18 | +1 |
| confidence>=0.9 会话 | 18 | 20 | +2 |
| confidence mean | 0.9096 | 0.9296 | +0.02 |
| confidence stdev | 0.0768 | 0.0593 | -0.0175 |

## 候选明细

### 新批次：图片误读（明确称图未提供）
无自动命中。

### 新批次：map 缺坐标/可靠性抱怨（不是未投送）
- `autosplice__58610__authentic__q95`: 未提供具体坐标，需根据 anomaly map

### 新批次：强相互验证（排除否定句）
- `autosplice__58610__authentic__q95`: 相互支持
- `genimage__SD14__045_sdv4_00039__q95`: 相互支持
- `autosplice__57501__0__q95`: 相互支持

### 新批次：被否定的相互验证句（不计为违规）
无自动命中。

### 新批次：视觉观察与 map 的相似措辞（非跨专家）
无自动命中。

### 新批次：弱一致措辞
- `genimage__Real__333c2446-4c1a-4c9f-b296-9c5b3c5fdf94__q95`: 结果一致
- `genimage__Glide__GLIDE_1000_200_00_048_glide_00127__q95`: 无冲突
- `autosplice__39890__1__q95`: 结果一致
- `autosplice__41187__1__q95`: 无冲突

### 新批次：笼统 Uncertain
无自动命中。

### 新批次：框文不一致
- `genimage__ADM__146_adm_91__q95`: 框中心在 [200.0, 300.0]，文字却提到 下方

### 新批次：文字提到异常区/高响应但无框
- `genimage__Real__333c2446-4c1a-4c9f-b296-9c5b3c5fdf94__q95`: 异常图中仅在图像中下部出现微弱黄色斑点（对应原始图像坐标约 [ymin=200, xmin=100, ymax=300, xmax=200]），但该区域在置信度图中为蓝色（低可靠性），说明该局部响应不可靠。综合判断，图像无明显局部编辑痕迹，且整图生成倾向为真实。两个专家的测量结果一致支持“Real”结论，且无冲突。
</reasoning>
<verdict
- `genimage__Real__351a8be1-104f-49be-9634-7c045dee48a2__q95`: OBE 的测量值（0.001153）极低，指向整图为真实图像（Real）。TruFor 的局部篡改分数（0.3396）低于其诊断阈值（0.5），且其异常图中红色高响应区域（疑似局部编辑候选）在可靠性图中对应蓝色低可靠性区域，表明该区域的检测结果不可靠。因此，无足够证据支持局部编辑存在。两个专家的测量结果均指向图像为真实且未被局部篡改，结论一致。
</reas
- `autosplice__42165__authentic__q95`: pulation_anomaly_score=0.382191，低于其诊断阈值 0.5，表明无强局部篡改迹象。其异常图显示旗帜区域有轻微红色响应，但置信度图显示该区域可靠性较低（蓝色边缘），不足以支持局部编辑结论。因此，局部编辑问题可判定为“Not tampered”，但需注意该结论基于当前测量，未排除微小编辑可能性。整图生成问题尚未测量，需调用 PROBE
- `genimage__Wukong__147_wukong_image128__q95`: 检测器，二者测量维度不同，不可互换或平均。TruFor 的高响应区域需结合其可靠性图判断，当前该区域可靠性低，不足以支持“局部篡改”结论。因此，局部编辑问题虽有测量，但证据强度不足，应标记为“Uncertain”。最终结论需基于PROBE的高分（AI生成）和TruFor的弱证据（局部编辑不确定）综合判断。
</reasoning>
<verdict>
{"v

### 全图 bbox 明细（人工核对这些框是否被写成局部定位）
- `genimage__SD14__045_sdv4_00039__q95` scope=global_range: TruFor异常图显示全图高响应，可靠性高，支持局部篡改嫌疑
- `autosplice__42005__0__q95` scope=global_range: TruFor anomaly map显示人脸区域高响应（红色），且confidence map显示该区域高可靠性（红色），提示局部编辑；PROBE高分支持整图AI生成。
- `autosplice__42005__2__q95` scope=global_range: TruFor anomaly map 显示人物头部轮廓区域（[0,1000] 原图坐标）呈现高响应（红色），对应 confidence map 在该区域显示高可靠性（红色），表明局部存在高置信度的篡改嫌疑。

