# 首轮 SFT 候选样板 — 待定判例问题

日期：2026-11-28。以下问题只涉及候选文本的判例选择，不改变已经跑出的工具分数、
evidence_id、三图或调用顺序。请审核时逐条决定“维持 / 改成什么 / 需要什么证据”。

## A. TruFor 高分真图：何时能写 Not tampered

候选里有 6 条 AutoSplice/Real 真图，TruFor 分数高于 0.5 或接近边界：

| sample | PROBE | TruFor | 异常图/可靠性图 | 候选 local | 待定问题 |
|---|---:|---:|---|---|---|
| `autosplice__51186__authentic__q95` | 0.0008 | 0.9723 | 高响应只在画框右缘窄条，可靠性高 | Not tampered | 画框边缘响应是否足以被明确排除？候选已判 Real/Not tampered，若认可，应成为“边缘响应”规则。 |
| `autosplice__65547__authentic__q95` | 0.0372 | 0.8972 | 高响应在右下肩部/衣领，可靠性中等偏高 | Uncertain | 原图是衣物褶皱与红墙交界，没有清楚内容替换边界。高分+内容区响应，是否仍只能 Uncertain？ |
| `autosplice__52769__authentic__q95` | 0.0937 | 0.5810 | 面部高响应，但同一面部区域可靠性低（蓝/青） | Uncertain | 低可靠区域是否直接排除，还是保留“候选但不可用”？候选按协议保留 Uncertain。 |
| `autosplice__57332__authentic__q95` | 0.4577 | 0.6269 | 仅画面右缘很小的高响应条，未落在枪械主体 | Uncertain | 分数过阈值但无内容区响应；应 Uncertain，还是可写 Not tampered？ |
| `autosplice__70684__authentic__q95` | 0.0022 | 0.5883 | 北极熊身体中部小范围高响应，可靠性较高 | Uncertain | 响应在内容上但占比小、无可见替换边界；是否可判 Not tampered？ |
| `autosplice__67006__authentic__q95` | 0.5527 | 0.4281 | 面部小范围高响应，总分低于阈值 | Uncertain | 整图 PROBE 仅边界高分 + TruFor 低于阈值但有小候选；整图/局部应各判什么？ |

核心判例：**“TruFor 分数高于阈值 + 响应落在内容上 + 原图没有清楚内容变化”** 时，
是维持 `local Uncertain`，还是允许 `Not tampered`？候选目前保守地保留 Uncertain；
51186 是唯一明确排除的例外（响应在画框边缘，不在内容上）。

## B. PROBE 高分但实际是局部编辑：整图维度怎么写

8 张 AutoSplice 篡改图的候选统一写成 `whole_image_generation=Real`、
`local_manipulation=Tampered`、总体 `Fake`；其中 6 张 PROBE 高于 0.5（45749、
69868、48251、72548、64223、45914），最高 0.9994。

待定问题：

1. 这些图里 PROBE 高分是否应被解释为“局部生成对象导致的整图响应”，从而维持
   whole Real？还是应把 whole 写成 Uncertain，避免用视觉判断推翻整图分数？
2. 如果维持 whole Real，报告里应如何写限制，才不会被读成“无视 PROBE”？
   候选目前写“PROBE 是整图分数、不定位；原图其余场景保持照片连续性”。
3. 对 Real `ac71afc5`（PROBE 0.5869、TruFor 0.2037、无局部篡改证据），候选把
   whole 写成 Uncertain；这与篡改图的 whole Real 是否矛盾？候选的区分是：
   篡改图有局部 Tampered 支撑总体 Fake，whole 只是来源维度；ac71 没有局部证据，
   PROBE 是唯一整图信号，所以 whole 保留 Uncertain。请确认这个区分是否成立。

## C. 调用顺序与“补测”

24 条都调用了两次专家，但顺序不同：13 条 TruFor→PROBE，11 条 PROBE→TruFor。
候选的理由是“先回答当前最不确定的问题，再补测另一个”。待定问题：

1. 对明显的局部候选（衣物、报纸、动物、人物），TruFor 先调用是否比固定 PROBE
   先调用更合理？
2. 对整图生成图（SD15/Wukong/Midjourney 等），先 PROBE 再 TruFor 是否只是
   “为了看局部图”，而不是必要补测？候选在 reasoning 里把 TruFor 定位为
   “确认局部响应是否只是整图生成的一部分”，请判断这是否足够。
3. 是否需要保留一条“只调用一个专家并说明为什么第二个不会改变结论”的样板？
   本轮为保证每条都有三图，24 条都调用了 TruFor；如果判例要求“该省则省”，
   下一轮应补一条单专家样板。

## D. 页面与审计

- `AUDIT_REPORT.md`：24/24 pass、0 问题（证据引用时序、数字出处、真值/mask 词、
  页面结构、链接、划分与组成）。
- `POST_RUN_PATCHES.md`：Midjourney 229 的 `0.39` 未投递数字已删除，改为定性读
  可靠性图；不影响工具输出/证据/结论。
- 审计是结构与出处检查，不判断“这条推理是否最优”；A/B/C 三组问题仍需人工判例。
