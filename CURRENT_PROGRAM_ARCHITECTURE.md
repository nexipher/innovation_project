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
  **冻结是硬约束**：`sft_data/FROZEN.json` 登记已冻结批次，`apply_review` / `build_final_v2` /
  `generate_trajectories_g4` 遇到即退出；分析用途加 `--allow-frozen`，产物标注
  `frozen_batch_used=true`。
  两条容易忘掉的规则写在台账顶部：
  **重新渲染工作台 ≠ 重新生成轨迹**；**自动检查通过 ≠ 允许训练**。

## 7. 数据布局

**两套布局，不混用**（2026-10-15 起）：

- `generation` 显式给出 ⇒ 新嵌套布局 `sft_data/<generation>/<kind>/<batch>/`
  （`kind` = trajectories / variants / review / train/batches）。
  批次名是**纯名字**（`b1`），代次是路径的一段，由调用方显式传入——没有"默认代次"。
- `generation` 省略 ⇒ 旧平铺布局（`sft_data/trajectories_<id>/` …），**只服务于历史复算**，
  且只对登记表里的历史 id 有效。三个写入入口（+ `build_variants_g4`）**必填 `--generation`**，
  所以新材料不可能落进平铺布局。

| 路径 | 内容 | 能否训练 |
|------|------|----------|
| `sft_data/<generation>/…/<batch>/` | 当前代次（g8）的批次产物 | 走完门禁才行 |
| `archive/legacy_v1_20261015/` | **v1 全部历史产物**（2026-10-15 归档，34 项 / 2141 文件） | 历史，不训练 |
| `archive/PATH_MAP.json` | 旧路径 → 归档路径；复算走 `utils.batch.resolve_path()`，命中即写 `resolved_from_archive` | — |
| `sft_data/FROZEN.json` | 冻结登记，身份是 **(代次, 批次)**；写入入口据此拒收 | — |
| `sft_data/GENERATIONS.json` | 代次索引（模型身份）；**批次身份仍以各批次 `VERSIONS.json` 为准** | — |
| `sft_data/review/dispositions.jsonl` | 人工处置台账（append-only，归档时留在原位） | — |
| `calibration/reliability_table.json` | 当前校准表（A 组拟合，逐格式权重）；运行时读者在阶段 4 归零后随归档移动 | — |
| `calibration/reliability_table_v1.json` | 修复前的表（旧批次复算用） | — |

**归档的使用规矩**：不删除、不覆盖；读归档路径必须记录 `resolved_from_archive`；
归档**不解除**冻结门禁；`train_lora_g4.py` 与 `finalize_sft_data.py` 对归档输入直接拒绝
（前者需 `--allow-archived`，且打印横幅）。清单见
`archive/legacy_v1_20261015/MANIFEST.json`，逐文件核对输出见操作日志。
