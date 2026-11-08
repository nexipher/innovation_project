# 2026-11-08 anomaly-peak 三图批次人工审核指引

> 状态：**pending human review**。同一 24 输入；本批使用异常峰值 inspection crop
> 修复“全图视图被抄成 `[0,0,1000,1000]`”的问题，并保留三图同区域/同内容尺寸、
> 框核验标记和分项证据覆盖检查。不是 SFT 数据。

## 重点核对

1. **全图框是否消失**：本批自动统计全图 bbox 为 0；看 `ADM` 的
   `location_text_mismatch` 是否只是文字/框中心边界问题。
2. **无 bbox 时的异常峰值裁剪**：`model_image_layout.region_source=anomaly_peak`；
   核对三图是否同内容尺寸、峰值是否在裁剪内、模型有没有把裁剪区域误当证据。
3. **证据覆盖纠正**：
   - `35b3b70e`：无 TruFor 时不得写 local `Not tampered`；
   - `39890_0/1`：无 PROBE 时整图分项应为 `Uncertain`，局部 Tampered 可保留；
   - `ADM`：初始整图 AI-generated 无 PROBE，反馈后是否补测。
   看 `coverage_check` 的 status / issues / feedback / original_output / corrected_output。
4. **局部误报**：`51136`、`ADM` 仍被判 local Tampered；`51136` 还没有局部框，
   需人工判断是否应拒收。
5. **confidence**：0.95 仍有 17/24；报告依据是否支撑。
6. **跨任务措辞**：本批自动剩 1 条强相互验证（`ADM`）；本批生成早于否定句作用域
   修复，下一次同批重跑应检查是否被纠正。

## 产物

- `index.html`：原始对话、三图投送、联合字段、框标记、coverage_check、调用/失败/停止。
- `comparison_vs_triptych_prefix/`：与前一轮全图框回归批次的比较。
- `comparison_vs_closing/`：与 `dual_20261108_closing` 的比较。
- `review_overlays/`：最终预测框叠加；全图框标为 `GLOBAL RANGE`。

自动统计不是验收结论；框/文字以 overlay 与原始对话为准。
