# 24 会话联合任务验收摘要（自动统计，非人工验收）

> **对照批次说明**：本批生成于 2026-11-07，当时仍使用机械联合规则；2026-11-08 已取消该规则，原 24 条保留为对照，不重生成。

> **状态：pending human review。** 本文只做描述统计，不能替代人工签字。

- 会话数：24
- 数据：GenImage_q95 12 / AutoSplice_q95 12；只用 dev/val。

### 总体

| 项 | 数量 |
| --- | ---: |
| 整图真值 Real | 16 |
| 整图真值 AI-generated | 8 |
| 局部真值 Not tampered | 8 |
| 局部真值 Not applicable | 8 |
| 局部真值 Tampered | 8 |

### 模型输出

| 项 | 数量 |
| --- | ---: |
| expert_calls=1 | 0 |
| expert_calls=2 | 24 |
| max images attached=4（TruFor 轮） | 23 |
| 有 failed_tool_calls | 1 |
| 有 joint_verdict_issues | 0 |
| 有 joint_overall_mismatch | 4 |
| 有 suspicious_regions | 10 |
| 区域总数 | 10 |
| 区域 evidence 为空 | 0 |
| 区域 bbox 覆盖全图 | 3 |

### 按真值分组的预测

| 真值 | 整图预测 | 局部预测 |
| --- | --- | --- |
| whole=Real | {'AI-generated': 8, 'Real': 8} | {'Not tampered': 1, 'Tampered': 6, 'Uncertain': 9} |
| whole=AI-generated | {'AI-generated': 8} | {'Tampered': 1, 'Uncertain': 7} |
| local=Not tampered | {'AI-generated': 3, 'Real': 5} | {'Not tampered': 1, 'Tampered': 1, 'Uncertain': 6} |
| local=Not applicable | {'AI-generated': 8} | {'Tampered': 1, 'Uncertain': 7} |
| local=Tampered | {'AI-generated': 5, 'Real': 3} | {'Tampered': 5, 'Uncertain': 3} |

### 停止原因

`{'model_concluded': 24}`

### 高 TruFor 响应真图（人工重点）

| 会话 | 真值 | 模型 verdict | whole | local | 区域数 | report（截断） |
| --- | --- | --- | --- | --- | ---: | --- |
| autosplice__58610__authentic__q95 | Real | Uncertain | Real | Uncertain | 0 | 图像整体为真实拍摄（PROBE分数0.144755，方向Real），但TruFor局部篡改检测分数0.65495高于阈值，提示可能存在局部编辑，但未提供具体可疑区域坐标及可靠性图，无法确认编辑位置，故局部操纵判断为Uncertain。 |
| autosplice__51136__authentic__q95 | Real | Fake | AI-generated | Uncertain | 1 | 图像被AI生成（PROBE分数0.9781），局部存在编辑候选（TruFor分数0.8137，区域[100,100,900,800]），但无明确篡改证据，故局部判断为Uncertain。 |
| autosplice__42165__authentic__q95 | Real | Real | Real | Uncertain | 0 | 整图判别器（PROBE）得分0.014967，指向真实拍摄；局部篡改检测器（TruFor）得分0.382191，低于阈值，未发现强局部异常。因TruFor未提供足够局部证据，局部状态定为Uncertain。无可疑区域。 |
| autosplice__53059__authentic__q95 | Real | Fake | AI-generated | Tampered | 1 | 图像经 PROBE 判定为 AI 生成（分数 0.967），TruFor 检测到局部篡改候选区域（分数 0.662，anomaly_map 高亮），结合两者证据，结论为伪造图像。 |

### 局部篡改真值（人工重点）

| 会话 | 会话真值 | whole | local | 区域数 | 失败 |
| --- | --- | --- | --- | ---: | --- |
| autosplice__39890__0__q95 | Fake/Tampered | AI-generated | Tampered | 1 | False |
| autosplice__39890__1__q95 | Fake/Tampered | AI-generated | Tampered | 1 | False |
| autosplice__41187__0__q95 | Fake/Tampered | Real | Uncertain | 0 | False |
| autosplice__41187__1__q95 | Fake/Tampered | Real | Uncertain | 0 | False |
| autosplice__72399__2__q95 | Fake/Tampered | AI-generated | Uncertain | 1 | False |
| autosplice__42005__0__q95 | Fake/Tampered | AI-generated | Tampered | 1 | False |
| autosplice__42005__2__q95 | Fake/Tampered | AI-generated | Tampered | 1 | False |
| autosplice__57501__0__q95 | Fake/Tampered | Real | Tampered | 1 | False |

### 工具失败

- `genimage__Midjourney__144_midjourney_34__q95`: ['trufor_expert: RuntimeError: trufor_expert：未产出 npz（退出码 0）\nstdout 尾部：=> loading model from /root/autodl-tmp/third_party/TruFor/pretrained_models/trufor.pth.tar\nEpoch: 81\n\nstderr 尾部：ch memory, this process has 1.51 GiB memory in use. Of the allocated memory 1.04 GiB is allocated by PyTorch, and 25.12 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation fo

## 说明

- `joint_overall_mismatch` 表示模型总体标签与两维推导不一致；未改写模型原文。
- `failed_tool_calls` 来自 trace metadata；本批至少有一例 TruFor 在 1024 图像上失败。
- 以上统计描述行为，不构成验收结论；SFT 与调用策略对照均未启动。
