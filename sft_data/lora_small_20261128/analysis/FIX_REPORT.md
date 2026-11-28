# 2026-11-28 局部漏检与结案修复 — 复验报告

状态：已完成（base/LoRA 各 30/30；同证据复验 10 条）。

## 结论摘要

- 三图从 anomaly-peak 半边长裁剪改为 high-response 包围盒；10 张篡改图的
  mask 覆盖率均值由 0.421 升到 0.787（57276 的孤立小峰回退旧裁剪；低响应/
  真图也保留旧 peak crop，避免无谓地切换成 full-frame）。
- 覆盖纠正会要求未测 TruFor 先补测；两位专家已测时不再诱使重复调用；
  57143 的“局部改 Uncertain 后总体也弃权”规则已改写。
- 30 图 before/after 指标见下。

## 基线（f0ab9ce）

| 指标 | 原版 | LoRA |
|---|---:|---:|
| 局部篡改检出（10） | 2 | 1 |
| 真图局部误报（10） | 0 | 0 |
| 真图整图误报（10） | 2 | 2 |
| 两分项同时判对（30） | 8 | 7 |
| 弃权（30） | 0 | 2 |
| 专家调用 | 54 | 49 |

## 修复后（两臂各 30/30 已完成）

| 指标 | 基线 base | 修复后 base | Δ | 基线 LoRA | 修复后 LoRA | Δ |
|---|---:|---:|---:|---:|---:|---:|
| 局部篡改检出（10） | 2 | 3 | +1 | 1 | 4 | +3 |
| 真图局部误报（10） | 0 | 0 | 0 | 0 | 0 | 0 |
| 真图整图误报（10） | 2 | 2 | 0 | 2 | 2 | 0 |
| 真图总体误报（10） | 2 | 2 | 0 | 2 | 2 | 0 |
| 两分项同时判对（30） | 8 | 12 | +4 | 7 | 14 | +7 |
| — 篡改图（10） | 1 | 0 | -1 | 1 | 1 | 0 |
| — 生成图（10） | 0 | 6 | +6 | 0 | 6 | +6 |
| — 真图（10） | 7 | 6 | -1 | 6 | 7 | +1 |
| 弃权（30） | 0 | 3 | +3 | 2 | 1 | -1 |
| 专家调用 | 54 | 58 | +4 | 49 | 56 | +7 |
| 分项/总体一致（30） | 30 | 29 | -1 | 28 | 30 | +2 |
| 生成图 local Not applicable（10） | 0 | 6 | +6 | 0 | 6 | +6 |
| 覆盖检查 failed / pass（当前检查器复算） | 3 / 27 | 1 / 29 | — | 3 / 27 | 2 / 28 | — |

- 局部检出：新增 43686、73275；57143 保持；42958 本轮被模型写成
  Uncertain（基线 Tampered），所以净 +1，不是 +3。
- 生成图 6 条 local Not applicable：ADM 860、BigGAN 739、Midjourney 191、
  SD14 705、SD15 998、VQDM 143；其中 6 条都同时判对整图/局部。
- 真图两边同时判对从 7 降到 6：a383 与 d1b49378 被写成 local Uncertain；
  65749/70102 仍是整图 AI 误报。真图局部 Tampered 误报仍为 0。
- 篡改图两分项同时对从 1 降到 0：42958 的 local 从 Tampered 变 Uncertain
  （高 TruFor 仍被模型保守化），且总体 Fake 与两个分项不一致；覆盖检查
  failed 的正是这条。其余 43686/57143/73275 的整图分项仍错（Uncertain 或
  AI-generated），所以即使局部正确也不能 both。
- base 30/30、1681.9 s、peak allocated 21.873 GB / reserved 23.868 GB；
  LoRA 30/30、2238.7 s、peak allocated 21.939 GB / reserved 24.522 GB。
- 覆盖检查 failed：base 1 条（42958，local Uncertain + overall Fake 不一致）；
  LoRA 2 条（52191 只用 PROBE、局部未测但标签本身正确；352 的跨任务措辞）。
  覆盖失败是审计记录，不是自动改写标签。
- LoRA 的局部检出从 1/10 升到 4/10（42958、43686、57143、73275），
  真图局部误报仍为 0；both 从 7/30 升到 14/30；一致性从 28/30 升到 30/30。
  57143 不再出现“局部 Uncertain → 总体 Uncertain”，保持 Tampered/Fake。
- 三图裁剪对编辑 mask 的覆盖率（`analysis/crop_coverage_fix.md`）：
  41993 0.369→0.965、43686 0.440→1.000、57143 0.520→0.981、
  73275 0.262/未调用→0.967、67065 未调用→1.000；40489 0.265→0.378、
  56086 0.679→0.768、60788 0.199→0.501、57276 维持 0.305（孤立小峰回退）。
  8 张有旧裁剪的均值 0.421→0.787；这是证据呈现指标，不等于局部判对。

## 57143_0 纠正轮复查

- 基线 base：TruFor 0.932、PROBE 0.981；初始 `AI-generated / Tampered / Fake`。
  旧覆盖提示把“非证明”当成“仅候选”，同时写“可以补测”，导致 base 在两位
  专家都已测量后又调用一次，触发 `loop_protection`；最终局部仍是 Tampered。
- 基线 LoRA：同样初始 `AI-generated / Tampered / Fake`；旧提示因“仅提供候选”
  要求局部改 Uncertain，随后又写“没有其他 Fake 分项时总体不能仍为 Fake”，
  模型把整图 AI-generated 这个 Fake 分项漏掉，改成 `AI-generated / Uncertain /
  Uncertain`；覆盖检查 failed，但最终输出仍被采用。
- 修复：已有 `suspicious_regions` 定位的 Tampered 不再因“仅候选/非证明”
  被降级；若局部仍需改 Uncertain，提示明确“whole=AI-generated 时总体保持
  Fake”；两位专家都已测量时禁止重复补测。
- 复验 base：`AI-generated / Tampered / Fake`，2 次调用（36.2 s），未再
  触发 loop_protection；局部检出保留。
- 复验 LoRA：`AI-generated / Tampered / Fake`，2 次调用（45.0 s）；
  局部 Tampered 保留，总体保持 Fake，不再出现“局部 Uncertain 后总体弃权”。

## 已看到的 base 变化（待全量）

> 本轮重跑中模型对同分证据的措辞仍有波动（例如 42958 这次把高 TruFor 写成
> Uncertain，而基线是 Tampered）；以下按本轮实际输出记录，不按预期改写。

- 40489：`AI-generated / Not tampered / Fake` → `AI-generated / Not applicable / Fake`；
  局部仍未检出（真值 Tampered），高响应裁剪覆盖率 0.378，属于 TruFor 定位弱。
  第二次窄纠正只把整图生成后的局部标签改为 Not applicable，不改变局部漏检。
- 41993：`AI-generated / Uncertain / Fake` → `Uncertain / Uncertain / Uncertain`；
  高响应裁剪覆盖率 0.966，但模型仍以“无具体内容变化”拒绝 Tampered，且整图
  变为 Uncertain；属于“证据可用但 Agent 漏判”。
- 42958：本轮 `Real / Uncertain / Fake`（基线 `Real / Tampered / Fake`），
  高 TruFor 0.999 仍被模型写成 Uncertain；这是模型保守判读的波动，未强行改写。
- 43686：`AI-generated / Uncertain / Fake` → `Uncertain / Tampered / Fake`；
  局部检出恢复，调用 2→1；原因是带 `suspicious_regions` 的 Tampered 不再被
  “仅提供候选”降级。
- 67065：`Real / Uncertain / Real` → `Real / Uncertain / Real`；补测后
  TruFor 0.991、高响应裁剪覆盖率 1.000，模型仍按“候选不是证明”写 Uncertain，
  属于“证据可用但 Agent 漏判”，未强行改写。
- 56086/57276/60788：TruFor 0.256/0.164/0.413，属于弱响应或定位偏离；
  裁剪覆盖率 0.768/0.305/0.501，局部仍未检出（56086 本轮 local=Not tampered），
  保留为专家限制。
- 73275：`AI-generated / Uncertain / Fake` → `AI-generated / Uncertain / Fake`；
  补测了 TruFor（调用 1→2、高响应裁剪覆盖率 0.967），但模型仍以“候选非证明”
  写 Uncertain；本次运行未恢复局部检出，保留为模型保守判读。

## 同证据复验（基线 trace 的同一 clean prefix）

同一原图、工具返回和三图，分别喂给原版/LoRA；不执行工具，不重跑专家。
9/10 条两臂标签完全相同，只有 SD14 705 的 local 不同（base Uncertain、
LoRA Not tampered）。

| sample | truth | base replay | LoRA replay |
|---|---|---|---|
| 41993 | Real/Tampered | AI / Uncertain / Fake | AI / Uncertain / Fake |
| 43686 | Real/Tampered | AI / Uncertain / Fake | AI / Uncertain / Fake |
| 57143 | Real/Tampered | AI / Tampered / Fake | AI / Tampered / Fake |
| 73275 | Real/Tampered | AI / Uncertain / Fake | AI / Uncertain / Fake |
| d3429f29 | Real/Not tampered | Real / Not tampered / Real | Real / Not tampered / Real |
| BigGAN 739 | AI/Not applicable | AI / Uncertain / Fake | AI / Uncertain / Fake |
| ADM 860 | AI/Not applicable | AI / Not tampered / Fake | AI / Not tampered / Fake |
| SD14 705 | AI/Not applicable | AI / Uncertain / Fake | AI / Not tampered / Fake |
| VQDM 143 | AI/Not applicable | AI / Not tampered / Fake | AI / Not tampered / Fake |
| Midjourney 487 | AI/Not applicable | Uncertain / Uncertain / Uncertain | Uncertain / Uncertain / Uncertain |

读法：
- 57143 的同一证据下两臂都能写 local Tampered；基线 LoRA 的丢失是**旧覆盖
  提示的纠正轮**造成的，不是 LoRA 读图差异。
- 41993、43686、73275 的同一证据复验两臂都写 local Uncertain；它们在自主流程
  中是否写 Tampered 取决于补测/纠正轮，而不是三图或模型本身。
- 生成图的 `Not applicable` 也不是同证据直接读出的：同一证据下两臂都写
  Not tampered/Uncertain；自主流程里是 `generated_whole_local_not_applicable`
  纠正和第二次窄纠正把标签改成 Not applicable。


## 遗留

- **模型保守判读未修**：41993（TruFor .816）和 67065（.991）补测后仍写
  local Uncertain；42958（.999）本轮也写成 Uncertain 并造成 overall Fake 与
  分项不一致。模型把高响应仍视为“候选非证明”，这不是补测提示能完全解决的。
- **生成图高 TruFor 假响应**：487（.910）和 Wukong（.983）仍写 local Tampered，
  真值应为 Not applicable；本轮的 generated 规则只在低分/无高响应时生效。
- **生成图 NA 的语义副作用**：40489、60788 是 AutoSplice 伪造图，但整图被
  PROBE 误判为 AI-generated，局部被改成 Not applicable；局部本来也未检出，
  但语义上把伪造图的局部问题抹掉了。
- **真图 local Uncertain**：a383、d1b49378 本轮被写成 Uncertain，导致 base 真图
  both 从 7 降到 6；真图局部 Tampered 误报仍为 0。
- **TruFor 弱/定位偏离**：40489 .444/IoU .127、56086 .256/.362、57276 .164/0、
  60788 .413/.340；高响应裁剪只能提高呈现覆盖率，不能修正响应位置本身。
  这些案例保留 TruFor 权重和阈值，单独报告。
- 30 图样本小；未重训、未扩 250 条、未碰 test、未改专家权重/阈值。
