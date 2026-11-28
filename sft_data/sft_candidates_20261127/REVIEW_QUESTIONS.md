# 首轮 SFT 候选样板 — 已确认判例与复看点

日期：2026-11-28。以下判例已按你的确认写入候选轨迹；工具输出、证据 ID、三图和调用顺序未改，只改候选文本、标签、certainty 和新增两条单专家样板。

## A. 整图与局部分开结案

已确认：PROBE 高分、局部证据充分，但整图来源仍缺独立支撑时，写
**整图 Uncertain、局部 Tampered、总体 Fake**。

| sample | PROBE | TruFor | 修订后 whole/local/overall | 说明 |
|---|---:|---:|---|---|
| `autosplice__54029__1__q95` | 0.7219 | 0.9995 | Uncertain / Tampered / Fake | 局部强；整图无独立来源支撑 |
| `autosplice__45749__1__q95` | 0.9994 | 0.9983 | Uncertain / Tampered / Fake | 企鹅；不再因背景像照片写整图 Real |
| `autosplice__69868__0__q95` | 0.8101 | 0.9952 | Uncertain / Tampered / Fake | 老虎；局部框较宽 |
| `autosplice__48251__0__q95` | 0.9308 | 0.9934 | Uncertain / Tampered / Fake | 人像；局部框较宽 |
| `autosplice__72548__1__q95` | 0.8643 | 0.9942 | Uncertain / Tampered / Fake | 舞者；定位精度有限 |
| `autosplice__64223__2__q95` | 0.5434 | 0.9934 | Uncertain / Tampered / Fake | PROBE 仅边界高分 |
| `autosplice__45914__2__q95` | 0.9899 | 0.9928 | Uncertain / Tampered / Fake | 公交车 |

**对照保留**：`autosplice__55310__1__q95`（PROBE 0.0782 低于阈值）保持
**整图 Real + 局部 Tampered**。Real 不是靠原图照片连续性推断，而是 PROBE 低分
提供了整图侧的支持；原图连续性只作为报告限制。

## B. 边缘响应与内容区异常

已确认：TruFor 高响应落在整图边缘、不在内容区时，不写局部 Tampered。

| sample | TruFor | 响应位置 | 修订后 local/overall/certainty |
|---|---:|---|---|
| `autosplice__51186__authentic__q95` | 0.9723 | 画框右缘窄条 | Not tampered / Real / **Medium** |
| `autosplice__57332__authentic__q95` | 0.6269 | 核对裁剪 [500,122,1000,494]：高响应在整图底边，另一小响应在整图右缘；不是内容区 | Not tampered / Real / **Medium** |
| `autosplice__65547__authentic__q95` | 0.8972 | 右下肩部/衣领，内容区，但无清楚替换边界 | Uncertain / Uncertain / null（保留） |
| `autosplice__52769__authentic__q95` | 0.5810 | 面部内容区，但同区域可靠性低 | Uncertain / Uncertain / null（保留） |

`51186`、`57332` 的 certainty 都是 Medium：结论有支持，但 TruFor 分数仍高于阈值，
边缘响应是残余限制。

## C. PROBE 调用理由

已统一：所有 PROBE 的 `Expert Target & Hypothesis` 只写整图来源判断，不再写
“补测以区分局部/整图”。TruFor 的调用理由仍可写局部异常、可靠性或内容解释。

## D. 两条单专家样板

用现有控制器重放，保留真实工具返回和停止理由：

| sample | 调用 | 候选结论 | 停止理由 |
|---|---|---|---|
| `single_probe__genimage__Glide__...` | PROBE only | Fake / AI-generated / Not applicable；整图证据充分，局部问题不适用 | `model_concluded` |
| `single_trufor__autosplice__54029__1__q95` | TruFor only | Fake / Uncertain / Tampered；局部证据充分，整图来源未测 | `model_concluded` |

两条都是 1 次专家调用、2 轮助手输出；不替代 24 条主样板，单独列在工作台
“单专家样板”区。

## E. 请复看时确认

1. 7 张 PROBE 高分的局部编辑是否都接受“整图 Uncertain、局部 Tampered、总体 Fake”；
   特别是 `64223`（PROBE 0.5434 仅略高于阈值）是否应与 `55310` 一样保留 Real。
2. `57332` 的边缘核对是否认可：裁剪布局 [500,122,1000,494]，高响应在整图底边；
   因此按 `51186` 同一口径判 Not tampered / Medium。
3. `51186` certainty 从 High 改 Medium 是否合适。
4. 两条单专家样板的停止理由是否符合“证据够就停”，以及是否需要把
   `single_probe` 的 PROBE 调用理由再扩展为 case-specific 整图假设。
5. 审计 `AUDIT_REPORT.md` 26/26 pass、0 问题；它检查出处/结构，不判断判例本身。
