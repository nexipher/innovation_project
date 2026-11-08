# 2026-11-08 certainty_v1 一致性复验批次人工审核指引

> 状态：**pending human review**。同一 24 输入；本批在 `dual_20261108_certainty`
> 之后加入 certainty 字段一次性纠错、总体-分项一致性检查、局部候选/Not applicable/
> 低分 Not tampered 检查，并把“Uncertain 与确定 Fake 分项不能并存”“PROBE/TruFor
> 逻辑独立”“未校准不等于无信息”“禁止交叉验证措辞”写入提示词。不是 SFT 数据。

## 重点核对

1. **certainty 字段**
   - 看 `certainty_check.status`：`pass` / `pass_after_correction` / `failed` /
     `not_corrected`。本批 23 pass、1 pass_after_correction、0 失败。
   - `verdict=Uncertain` 时 `certainty` 必须为 `null`；Real/Fake 时为
     High/Medium/Low。核对没有把 Low/Medium 静默转成 null，也没有改标签。
   - 本批 High 21/23；虽然 schema 合法，但需人工判断 High 是否仍机械。
2. **总体与分项一致**
   - 不允许 `verdict=Uncertain` 同时保留 `whole_image_generation=AI-generated`
     或 `local_manipulation=Tampered`；若保留 Uncertain，该分项必须降为 Uncertain。
   - 不允许总体 Real 同时保留确定 Fake 分项；也不允许总体 Real、PROBE 可用却
     完全未测量而仅凭局部“未发现篡改”结案。
   - `local_manipulation=Not applicable` 只在 `whole_image_generation=AI-generated`
     时合法；本批重点核对 `58610`（真实、局部候选）。
3. **关键案例**
   - `Midjourney`：TruFor 失败，PROBE 高分 → 应为 Fake/AI-generated/Uncertain。
   - `SD14`、`SD15`、`Wukong`：局部未测/低分不应把整图 AI 结论改成总体 Uncertain。
   - `35b3b70e`、`58610`、`333c2446`、`42165`：低分/无局部证据时应为 Real +
     Not tampered；看是否被过度降为 Uncertain。
   - `339e8fae`、`51136`、`53059`：真实图被 PROBE 高分带成 Fake；局部是否保持
     Uncertain（不再误写 Tampered）。
   - `39890_0/1`、`41187_0/1`、`42005_0/2`、`72399_2`、`57501_0`：真实局部篡改
     案例；核对哪些被保守写成 Uncertain/Not tampered（本批局部召回 7/24）。
   - `ADM`：整图 Uncertain、local Tampered；真值 local Not applicable，需人工判断。
4. **覆盖纠正**
   - 看 `coverage_check.status`、`issues`、`remaining_issues`；本批 3 条重答、
     0 失败，仍要核对纠正后的标签是否与报告一致。
5. **测量噪声**
   - `failed_experts` 中 TruFor “未产出 npz” 是外部失败，不是中性测量；本批
     `Midjourney`、`42005_2` 受影响。
6. **三图/坐标/措辞**
   - 三图同区域、同内容尺寸、框核验标记；全图 bbox 不得作为局部框。
   - 不得写“交叉验证/相互验证/共同支持/相互支持”；只能分别说明 PROBE 的整图
     贡献和 TruFor 的局部贡献。

## 产物

- `index.html`：原始对话、三图投送、联合字段、certainty/coverage 检查、调用/失败/停止。
- `comparison_vs_certainty/`、`comparison_vs_peakcrop/`：描述性比较。
- `review_overlays/`：最终预测框叠加；全图框标为 `GLOBAL RANGE`。

自动统计不是验收结论；框/文字以 overlay 与原始对话为准。
