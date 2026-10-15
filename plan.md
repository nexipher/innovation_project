# 计划（当前代次 g8）

> **本文件从 2026-10-15 起只记当前代次的工作，只追加、不改写已写下的条目。**
> v1 计划（Qwen2.5-VL + PROBE/noise/jpeg/frequency_v2 时代，含其全部阶段与决策记录）已归档：
> `archive/legacy_v1_20261015/docs/plan_v1.md`，配套操作日志
> `archive/legacy_v1_20261015/docs/claude_operation_log_v1.md`。
> 归档索引：`archive/legacy_v1_20261015/MANIFEST.json`；旧路径映射：`archive/PATH_MAP.json`。
>
> 上位文档：`NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md`（新专家组与 Agent 重整方案）、
> `NEW_GENERATION_NAMING.md`（代次/冻结/归档映射的实施细节）、
> `CURRENT_PROGRAM_ARCHITECTURE.md`（当前架构）、`KNOWN_ISSUES.md`（问题清单）、
> `TESTING.md`（测试分层与运行方式）。

## 0. 当前状态（2026-10-15）

| 事实 | 值 |
|------|-----|
| 当前代次 | `g8`（`sft_data/g8/`，模型 Qwen3-VL-8B **待确认版本**；`g7` 及更早已冻结） |
| 默认策略 | `v3`（Qwen 主导；`config.HALTING_POLICY`，可用 `FORENSIC_HALTING_POLICY` 选择 v1/v2 复算） |
| 数据布局 | 代次是目录、批次名是纯名字：`sft_data/<generation>/{trajectories,variants,review,train/batches}/<batch>/`；三个写入入口的 `--generation` **必填** |
| 冻结 | `sft_data/FROZEN.json` 以 **(代次, 批次)** 为身份，代码层拒收；**两级**：未登记代次一律拒、冻结代次下不得新建任何批次、已登记冻结批次走 `--allow-frozen` 复算。九个写入入口同一函数 |
| 归档 | v1 全部产物已移入 `archive/legacy_v1_20261015/`（34 项、2141 个文件），逐文件核对通过 |
| 上一批实测 | `g7_b2`（56 会话，v3）：协议合规 56/56、0 违规、0 解析失败；但 **PROBE 单专家 53/55 优于 Agent 44/56** —— 这是换代的直接依据，不是待修的小问题 |

## 1. 阶段 1（**已完成，2026-10-15**）：目录隔离、冻结升级、归档

清单见 `NEW_GENERATION_NAMING.md` §6。执行记录（每一步做了什么、核对了什么）见
`claude_operation_log.md` 2026-10-15 的两条。当前状态：

- [x] 1. `utils/batch.py`：`batch_paths(batch, generation=None)` 支持嵌套布局；`assert_not_frozen_batch` 要求显式代次；`frozen_batches()` 返回 `{(generation, batch): entry}`；`resolve_path()` 查 `archive/PATH_MAP.json`
- [x] 2. `sft_data/FROZEN.json` 加 `generations` 索引；新建 `sft_data/GENERATIONS.json`；建 `sft_data/g8/` 空骨架
- [x] 3. 写入入口加**必填** `--generation`：`generate_trajectories_g4` / `build_final_v2` / `apply_review` 之后，逐次补齐 `build_variants_g4`、`admit_trajectories_g4`、`build_review_workstation`、`record_review`、`build_review_from_trajectories`、`g7_select_sources`——**共九个**，清单写在测试里（`tests/test_batch_binding.py` 的 `WRITERS`），漏一个就红
- [x] 3b. 冻结检查升级为**两级**（`assert_generation_is_writable`）：未登记代次一律拒（拼错代次名无 `--allow`）；冻结代次下**不得新建任何批次**（换个没登记的批次名也不行）；已登记的冻结批次仍可用 `--allow-frozen` 做带标记的复算
- [x] 4. 归档移动：docs+history 34 项 → `archive/legacy_v1_20261015/{docs,experiments,review}/`；**代码一个都没动**
- [x] 5. 写 `MANIFEST.json`（四段结构 + 留在原位清单 + 使用限制）、`PATH_MAP.json`、`NOTE.md`
- [x] 6. 移动前后摘要核对：`archive_inventory.py --verify` 通过；另加 `archive_v1.py --verify` 逐文件对 `HEAD` blob 核对（含 438 页登记过的链接改写）
- [x] 7. 路径修正：历史复算脚本走 `resolve_path` 并在报告里写 `resolved_from_archive`；训练/出集脚本拒绝读取归档的 v1 数据（`--allow-archived` 才放行，且打印横幅）
- [x] 8. 新版 `plan.md`、`claude_operation_log.md` 建立，顶部指向归档历史
- [x] 9. README 统一改写：删旧字段，只留当前能跑的内容 + 指向归档的一句话；`CURRENT_PROGRAM_ARCHITECTURE.md` §7 数据布局同步
- [x] 10. 测试路径修正并全绿（core 348 / module 919 / full 1157 passed）→ 提交本阶段

**本阶段额外发现（见操作日志）**：`resolve_path` 对单文件条目的尾斜杠 bug；`batch_inventory`
在归档后再次出现"无导出文件"假零；`g7_label_transitions` 与若干测试因空读而静默退化为零/跳过。

**本阶段不做**：不搬任何代码文件（`experts/frequency.py` 等退役实现仍被包导出与测试导入，退役放到阶段 4 与导入方一起改）；不解除任何冻结门禁；不建新版运行时校准表。

## 2. 阶段 2：Qwen3-VL 迁移冒烟（未开始）

入口 `scripts/g8_smoke.py`（待写），产物写 `sft_data/g8_smoke/`，不进任何批次目录。逐项断言：
图像确实进入多模态消息（不是只给路径）；一次合法调用被执行；结果进入正确的后续消息且被准确引用；结案可解析；
**模型原始回复、最终采用回复、外部控制原因三者分别留存**。CPU 用 `MockMLLMClient` 先跑通协议，GPU 换 Qwen3-VL（**另行授权**）。

## 3. 阶段 3：替代专家评测（未开始）

**保留集以提案为准**（`NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md` §1）：

| 专家 | 处置 |
|------|------|
| PROBE | **保留**，主检测器（C 组独立验证过） |
| noise | **候选保留**：与 NPR/UnivFD 同标准评测，**仅在缺乏独立效果与互补性时退出**——不预设退役 |
| NPR / UnivFD | 候选补位 |
| frequency_v2、jpeg | **已决定退出**当前工具箱 |

四专家是初始方案、不是数量硬约束。判据是**在 Agent 流程里的增益与互补性**，
不是单专家准确率。**前置事实待用户提供**（权重、许可、输入接口、输出定义）。

## 4. 阶段 4：解耦与退役（未开始）

- 移除 v3 路径上的参考后验与预计收益计算（含方向检查改按**经核实的原始 score 语义**）
- 退役 **frequency_v2 与 jpeg**（提案 §1 已决定）；**noise 的退役不是既定项**——它由阶段 3
  的评测结果决定，评测未出结果前不写进退役名单
- 顺序：先改导入方与包导出，再移动实现；`calibration/` 随之归档
- README/文档同步；旧校准表的运行时读者归零后才归档

## 5. SFT（未开始）

在新代次样本上做；审核工作台按批次隔离；训练入口沿用冻结门禁与归档拒收。

## 6. 硬性约束（任何阶段都适用）

1. **GPU 操作需事先明确授权**；核心逻辑必须能在 CPU 上跑通并测试。
2. **不推送远端**（本机到远端网络不可靠，只本地提交）。
3. **不复制、不重分发 PROBE / TruFor 的代码与权重**。
4. **不删除数据**（未经确认）；归档一律"移动 + 摘要 + 映射"，不覆盖、不改写。
5. 文件驱动流程：本文件先于代码、只追加；`claude_operation_log.md` 每条六字段；
   每个大段落更新 README；提交信息写清动机。

## 7. 等用户提供的事实（阻塞阶段 2–3）

1. Qwen3-VL-8B 的**具体 checkpoint**、推理模式（thinking / non-thinking）、许可与显存要求
2. NPR / UnivFD 的**权重来源、许可、输入接口、输出定义**
3. noise 专家的**评测口径与保留判据**（它是候选，退役与否由阶段 3 的增益/互补性结果决定）
4. 重复请求轮的处置口径：**掩码**该轮，还是**整条会话拒收**
