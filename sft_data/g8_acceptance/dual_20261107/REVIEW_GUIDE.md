# 人工验收指引：PROBE + TruFor 联合任务

> 状态：**pending human review**。本目录是验收材料，不是 SFT 数据；自动检查不能替代人工签字。
> 入口：`index.html`；逐会话页展示原始对话、实际投送图片（原图/区域/异常图/置信图）、
> 联合字段、可疑区域、调用计数和停止原因。

## 批次

固定 **24** 个会话，只用 dev/val，不碰 test：

| 类别 | 数量 | 说明 |
| --- | ---: | --- |
| GenImage Real | 4 | 普通真图（PROBE 可能误报） |
| AutoSplice Real | 4 | TruFor 真图高响应样例（`trufor_real_high_response`） |
| GenImage AI-generated | 8 | ADM/BigGAN/Glide/Midjourney/SD14/SD15/VQDM/Wukong 各一 |
| AutoSplice 局部篡改 | 8 | 含同源多变体（7 条 `same_base_multi_variant`） |

## 每条会话要核对

1. **工具理解**：PROBE 只回答整图生成；TruFor 只回答局部篡改。模型是否把 TruFor
   的分数说成整图生成概率，或把 PROBE 的分数说成局部区域？
2. **工具选择**：模型是否在需要局部判断时调用 TruFor？是否只因为已有 PROBE 就跳过?
   是否重复调用、浪费轮次？
3. **图是否真进入输入**：逐轮页应显示实际投送的异常图和置信图；调用日志里的
   `images_attached` 应随 TruFor 轮增加（原图 + 三张产物 = 4）。
4. **怀疑与修正**：首轮 `Visual Observation` 是否具体？后续是否根据 map/conf
   修正或保持怀疑？有没有把亮区直接认定为篡改、忽略低置信度？
5. **联合答案**：`whole_image_generation`、`local_manipulation`、
   `suspicious_regions` 是否分别填写；区域是否有 `[0,1000]` bbox、原因和证据 id；
   `verdict` 与两维推导不一致时是否被 `joint_overall_mismatch` 标出。总体 `Fake`
   可能来自任一维（甚至一维误报、另一维正确），必须拆开核对，不能只看总体标签。
6. **解释忠实**：`report` 是否引用了模型实际收到的数值/图像，而不是编造证据；
   `primary_evidence` 是否对应真实 evidence id；失败/缺失是否被说成中性结果。

## 明确不做

- 不把本目录当训练集；不运行 SFT、调用策略对照或阈值调整。
- 不以“自动检查通过”替代人工验收；不因个别模型错误改写 trace。
- 完成人工 dispositions 后，才讨论单专家/固定双专家/自主调用对照与 SFT。
