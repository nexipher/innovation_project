# g8 200-source development evaluation — 人工复核指引

> 自动统计不是验收结论；`index.html` 和原始 trace 是权威。

## 优先复核顺序

1. 全部总体错误、弃权、未形成结论。
2. 全部工具失败和 coverage/certainty 纠正案例。
3. 固定随机抽查约 20 条“总体正确且 certainty=High”的会话，检查结论正确但解释错误。

固定队列见 `review_queue.json`（156 条）；每条包含 reasons。

## 重点

- 总体误判：真实图被 PROBE 高分带成 Fake；局部篡改被保守写成 Uncertain。
- 局部任务：AutoSplice 的 Tampered/Not tampered/Uncertain/未调用/失败分开看；
  GenImage 纯生成图的 local Not applicable 按单独口径看。
- 三图同位置、可靠性图背景红是否被误套、框文矛盾、跨任务相互验证措辞、编辑过程断言。
- 报告的主要限制是否与 certainty 协调；High 是否机械集中。
- 区域定位只在使用对应 mask 时评价；本批 AutoSplice mask 为 base 共享，且每源只选一个 variant。

## 导出

- `records.json` / `review_records.csv`：逐样本结论与审计字段。
- `SUMMARY.md`：自动统计。
- `index.html`：逐会话工作台。
