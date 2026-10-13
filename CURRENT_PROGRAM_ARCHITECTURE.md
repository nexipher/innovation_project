# 当前程序架构（只描述实际代码）

> 本文只写**现在能运行什么、由谁决定什么**。历史方案、被否掉的路线、实验记录在
> `plan.md` 与 `claude_operation_log.md` 里，不再是当前指令。最后核对时间：2026-10-13。

## 1. 当前任务

**全局生成检测**：判断一张图是相机拍摄（Real）还是扩散/生成模型产出（Fake）。

- **不做**局部篡改定位、不做伪造区域分割——现有专家的测量范围都是
  `measurement_scope="global"`（整图），bbox 只是诊断关注点，不限定测量。
- 专家给出的是**统计证据**；结论由模型综合（见下）。

## 2. 谁决定什么

| 角色 | 决定 | 不决定 |
|------|------|--------|
| **Qwen（v3 策略）** | 调用哪个专家、何时结案、**结案标签与自报置信度** | 不决定能否调用（预算与协议由控制器把关） |
| **控制器** | 执行调用、单动作协议、重复调用跳过、预算/防循环停止、证据账本检查与一次纠正、门禁 | 不决定标签（v3 下） |
| **参考后验（策略计算）** | 记录在案，供对照 | **不覆盖模型标签**（`reference.recommendation` 只是参考） |

v3 的停止原因各自独立记录：`model_concluded`（模型自主结案，标签即其结案）、
`budget_exhausted` / `loop_protection` / `format_failure`（外部截断，结案由收尾轮给出，
不可解析则记 `no_parseable_conclusion`——**这不是弃权**）、`evidence_check_failed`。

## 3. 唯一推荐启动命令

```bash
# 当前任务（全局生成检测，Qwen 主导）——单张图像，人在回路外
FORENSIC_HALTING_POLICY=v3 python main.py --image <path/to/image>
```

**默认配置就是当前版本 v3**（`config.HALTING_POLICY`）；历史复算才需要显式
`FORENSIC_HALTING_POLICY=v2`（或 `v1`）。报告与批次必须核对三个版本：策略、提示词指纹、
校准表身份（`utils/config_fingerprint`，也写进每个批次的 `VERSIONS.json`）。

批量与验收（不是"启动命令"，是流程步骤）：

```bash
python scripts/generate_trajectories_g4.py --batch <id> --policies probe+noise   # GPU
python scripts/g7_b2_report.py --batch <id>                                       # 统计
python scripts/cpu_acceptance_r5.py                                               # CPU 五检查
python scripts/build_review_from_trajectories.py --batch <id>                     # 审核材料
python scripts/batch_inventory.py                                                 # 批次台账
```

历史版本只能显式开启：`FORENSIC_HALTING_POLICY=v1|v2`（非法值直接报错）；
无环境变量时即当前版本 v3。

## 4. 实际可用的专家

| 工具键 | 代码 | 权重 | 当前角色 | 备注 |
|--------|------|------|----------|------|
| `probe` | `experts/probe_dino.py` | **外部，不随仓库分发**（`PROBE_WEIGHTS`） | 主检测器（分离度 ≈0.98，A 组拟合） | 未配置权重时该工具**缺席**，其余照常 |
| `noise` | `experts/noise.py` | 仓库内 | 辅助线索；高值统计上指向 Real（极性已校正） | 计划退出工具箱 |
| `jpeg` | `experts/jpeg.py` | 仓库内 | 辅助线索；q70 下分离度仅 0.40 | 计划退出工具箱 |
| `freq` | `experts/frequency_v2.py` | 仓库内 | 弱信号（0.66），只作辅助 | 计划退出工具箱 |
| — | `experts/frequency.py`（v1）、`experts/ela.py` | — | **已退役**，不进入工具箱 | 保留代码供复算 |
| — | `experts/trufor.py` | 外部 | **不在全局主链**，只作对照评测 | 与 PROBE 同属外部模型，未经许可不得再分发 |

工具集会按格式适用性裁剪（`APPLICABILITY_THRESHOLD=0.65`）：不适用的组合直接不生成该条会话。

## 5. 运行时序（一次会话）

```
读图 → 系统提示（SOP：一轮一个动作；bbox 可选；先描述后判定）
  ↓ 循环（≤MAX_MODEL_TURNS=6，专家调用 ≤MAX_EXPERT_CALLS=5）
模型输出 → Parser.protocol_action（论述块内的标签不计为请求）
   ├ 违规（调用+结案 / 两个调用）→ 不执行任何动作，发协议纠正
   ├ 合法调用 → 已测量过则跳过并说明（重复请求计为无进展）；
   │            否则测量 → 注入完整 token → 追加“本会话状态”行
   └ 合法结案 → 进入结案流程
结案流程（所有路径共用）：
  账本检查（专家/字段/数值/方向）→ 有缺陷则一次纠正 → 仍不合格记 failed
  → 落 final_verdict：标签、verdict_status、证据检查、参考后验（单列）
```

## 6. 版本记录与门禁

- **配置指纹**（`utils/config_fingerprint`）：策略、提示词、专家规格（含权重摘要）、
  校准表身份、rectifier、tokenizer、git commit → 每个批次目录写 `VERSIONS.json`。
- **批次台账**：`sft_data/BATCH_INVENTORY.md`（`scripts/batch_inventory.py` 生成），
  逐批写明类别（历史/待审核/合格训练）与三个版本。
- **进 SFT 的唯一路径**：`record_review --import` → `apply_review` → `build_final_v2`（门禁）。
  两条容易忘掉的规则写在台账顶部：
  **重新渲染工作台 ≠ 重新生成轨迹**；**自动检查通过 ≠ 允许训练**。

## 7. 数据布局

| 路径 | 内容 | 能否训练 |
|------|------|----------|
| `sft_data/train/final/`、`final_v2/` | 冻结的历史基线（v1/v2 时代） | 只作对照 |
| **过去的全部批次**（含 g7_b1、g7_b2、237 集与缺陷归档） | **2026-10-13 起全部冻结为历史实验**：更换专家后这些测量系统将不再存在 | 一律不得用于 SFT；新材料由新版专家生成 |
| `sft_data/train/batches/<id>/` | 分批产物；`review_only.json` = 评审材料 | 仅当有审核处置且过门禁 |
| `sft_data/trajectories_<id>/` | 批次轨迹（只读，含 `VERSIONS.json`） | 不直接训练 |
| `sft_data/trajectories_<id>_<原因>/` | 留证归档（缺陷作废批次） | 不用于任何统计或训练 |
| `sft_data/review/<id>/` | 工作台页面（可重渲染） | — |
| `calibration/reliability_table.json` | 当前校准表（A 组拟合，逐格式权重） | — |
| `calibration/reliability_table_v1.json` | 修复前的表（旧批次复算用） | — |
