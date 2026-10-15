# v1 归档清单（预检；本文件由 `scripts/archive_inventory.py` 生成）

生成时间：2026-10-15T08:42:17｜**未移动、未删除任何文件**

**这张表能证明什么**：路径存在、大小、文件数、内容摘要（大目录只覆盖前 512 MB，已逐行注明）、是否被 git 跟踪。**不能证明什么**：归档目标位置是否足够、移动后历史复算是否仍可跑（需要路径映射）、以及完整性——完整性要用这张表在移动**之后**复核。

**`role` 的含义**：`runtime` 表示当前代码会读它，**不能盲目搬走**（提案 §2.6）；`history` 是产物；`docs` 是决策记录；`external` 在仓库之外。

| 路径 | 角色 | 类型 | 文件数 | 大小 | 摘要 | git | 说明 |
|---|---|---|---|---|---|---|---|
| `plan.md` | docs | file | — | 207 KB | b0b2b382f2f80e8b | yes | 计划与档案；顶部当前阶段一并封存 |
| `claude_operation_log.md` | docs | file | — | 271 KB | b6630388b0d55842 | yes | 操作日志（增量追加） |
| `CLAUDE_CODE_G7_REPAIR_HANDOFF.md` | docs | file | — | 14 KB | 24adcc36641af761 | yes | G7 修复交接文档 |
| `NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md` | docs | file | — | 12 KB | a0cb6e5a42fb1582 | yes | 本方案本身（随代次留存，不归档） |
| `KNOWN_ISSUES.md` | docs | file | — | 16 KB | b9a068bb3f325886 | yes | 问题清单（新版由 `KNOWN_ISSUES.md` 重写） |
| `sft_data/trajectories` | history | dir | 440 | 1.6 MB | 09f4b20e2e180a89 | yes | 237 集轨迹（v1/v2 时代） |
| `sft_data/trajectories_g7_b1` | history | dir | 269 | 954 KB | a0e209d2b947ed17 | yes | b1 轨迹 + VERSIONS.json |
| `sft_data/trajectories_g7_b2` | history | dir | 61 | 310 KB | 4aa50568ff820f03 | yes | b2 轨迹 + VERSIONS.json |
| `sft_data/trajectories_g7_b2_parser_bug` | history | dir | 57 | 208 KB | e76a88907c1f652c | yes | 缺陷留证归档 |
| `sft_data/trajectories_g7_b2_reasoning_prose` | history | dir | 10 | 56 KB | c21107c6974fb587 | yes | 缺陷留证归档 |
| `sft_data/trajectories_g7_b2_statefix` | history | dir | 0 | 0 KB | e3b0c44298fc1c14 | no | 状态行对照素材 |
| `sft_data/variants` | history | dir | 113 | 14.1 MB | 207857585dea50ef | yes | 237 集变体图 |
| `sft_data/variants_g7_b1` | history | dir | 354 | 38.8 MB | 183627032c9085b5 | yes | b1 变体图 + 清单 |
| `sft_data/variants_g7_b2` | history | dir | 354 | 41.3 MB | 15d11b8ef13069b6 | yes | b2 变体图 + 清单 |
| `sft_data/train/final` | history | dir | 6 | 2.4 MB | 71c97d212c51582b | yes | 冻结基线 v1 |
| `sft_data/train/final_v2` | history | dir | 5 | 1.9 MB | c309842529289433 | yes | G4 转换产物（237 集） |
| `sft_data/train/batches/g7_b1` | history | dir | 8 | 1.7 MB | 13c77d71d222a0ae | yes | b1 评审样本 + 清单 |
| `sft_data/train/batches/g7_b2` | history | dir | 3 | 858 KB | 1933adf70f3637f1 | yes | b2 评审样本 + 清单 |
| `sft_data/train/sft_borderline.json` | history | file | — | 0 KB | 4f53cda18c2baa0c | yes | 早期分桶样本（correct/conflict/format/…） |
| `sft_data/train/sft_conflict.json` | history | file | — | 493 KB | 7a1e3bbddb07866b | yes | 早期分桶样本（correct/conflict/format/…） |
| `sft_data/train/sft_correct.json` | history | file | — | 502 KB | e0e6d38841822ec4 | yes | 早期分桶样本（correct/conflict/format/…） |
| `sft_data/train/sft_format.json` | history | file | — | 0 KB | 4f53cda18c2baa0c | yes | 早期分桶样本（correct/conflict/format/…） |
| `sft_data/train/sft_rejected.json` | history | file | — | 142 KB | b3c52b55537d1eca | yes | 早期分桶样本（correct/conflict/format/…） |
| `sft_data/review` | history | dir | 450 | 12.5 MB | da757a53148ca6d6 | yes | 工作台页面、人工审核 JSON、dispositions |
| `sft_data/stream_a_normal` | history | dir | 0 | 0 KB | e3b0c44298fc1c14 | no | 早期 SFT 数据流 |
| `sft_data/stream_b_conflict` | history | dir | 0 | 0 KB | e3b0c44298fc1c14 | no | 早期 SFT 数据流 |
| `sft_data/admission_report.json` | history | file | — | 317 KB | 73dab6fc9e4ca33b | yes | 237 集准入报告 |
| `sft_data/admission_report_g7_b1.json` | history | file | — | 139 KB | dbe8689f6c1492ff | yes | b1 准入报告 |
| `sft_data/metadata.json` | history | file | — | 0 KB | cbc09c5f46d2035b | yes | 早期数据元信息 |
| `sft_data/trajectories_report.json` | history | file | — | 8 KB | 62b886aefb575c99 | yes | 早期运行报告 |
| `calibration` | runtime | dir | 2714 | 1463.4 MB | 94aafbad1c722855（部分） | yes | 校准表（`reliability_table.json` 是**运行时默认路径**）与历史审计报告混居；须先分离再归档 |
| `calibration/set` | runtime | dir | 1396 | 261.2 MB | 04a1032c9b7de6ec | yes | 划分与原始分数（构建脚本的输入） |
| `calibration/reliability_table_v1.json` | runtime | file | — | 8 KB | d08051253a8278c3 | yes | 旧表，供历史复算；新版不读 |
| `sft_data/split_v2.json` | runtime | file | — | 2.1 MB | d89de39aa95b95b8 | yes | 数据划分（运行与构建都读） |
| `sft_data/FROZEN.json` | runtime | file | — | 1 KB | 5f5d91391a89d560 | yes | 冻结登记：**归档不等于取消门禁**（提案 §2.7） |
| `experts/frequency.py` | code | file | — | 9 KB | 85b887d6f725f14d | yes | 退役专家 v1 实现，**仍被 experts/__init__ 与多个脚本/测试导入**（被 6 处导入：experts/__init__.py、scripts/calibrate_experts.py、scripts/evaluate_experts_g2.py、tests/conftest.py…） |
| `experts/ela.py` | code | file | — | 4 KB | bacc42735d147076 | yes | 退役专家实现（同样按代码退役处理，不随数据归档移动）（被 1 处导入：scripts/evaluate_experts_g2.py） |
| `experts/trufor.py` | runtime | file | — | 24 KB | 4406f0592b29d688 | yes | 外部对照模型（不进主链，但不属本代次归档对象） |
| `scripts/evaluate_experts_g2.py` | runtime | file | — | 21 KB | 28afdd2efaa0d0da | yes | 旧专家的评测脚本；退出后按退役归档 |
| `scripts/calibrate_experts.py` | runtime | file | — | 14 KB | 230ff21201f0a53a | yes | 旧校准流程；新代次沿用或重写前不能移 |
| `scripts/build_reliability_table.py` | runtime | file | — | 17 KB | 3a5070b6a0f5eede | yes | 校准表构建；新专家接入要改这个 |
| `utils/reliability.py` | runtime | file | — | 5 KB | b32fbd0449d90ced | yes | 运行时读取校准表；旧数值依赖的解除点 |
| `dataset` | runtime | dir | 9000 | 2871.6 MB | 29f6ebb06876e24a（部分） | no | 数据源，继续使用（提案 §2.3）（体积大：摘要只覆盖前 512 MB 的名称与字节数，内容未逐块校验） |
| `models` | external | 缺失 | — | — | — | no | 仓库内模型权重目录（若存在） |
| `/root/autodl-tmp/third_party/probe` | external | dir | 2 | 1161.3 MB | c4a9baf224f291f7（部分） | no | PROBE 权重（不随仓库分发）（体积大：摘要只覆盖前 512 MB 的名称与字节数，内容未逐块校验） |
| `/root/autodl-tmp/third_party/TruFor` | external | dir | 160 | 592.0 MB | 643a5b6ad05be361（部分） | no | TruFor 权重与代码 |

## 总量（先把口径说清楚）

- **待归档（role=docs+history，已去掉父子重叠）**：30 项，118.7 MB
- **保持原位（runtime/code/external/dataset）**：15 项，6351.6 MB

两个数字都**不是“磁盘总占用”**：清单里既有父子目录重叠（如 `calibration` 与其子目录），也有仓库之外的外部路径，还有本就不归档的 dataset 与权重。**大目录的部分摘要不足以证明完整备份**——真正移动之前，对每个待归档目录做一次全量摘要（本清单已对 docs/history 做全量），移动之后再用 `--verify` 逐行复核。

## 归档与退役是两件事

`role=code` 的项是**仍被导入的已退役实现**（如 `experts/frequency.py` 仍被 `experts/__init__.py`、评测脚本与测试导入）。**数据归档不动代码**；代码退役在第 4 阶段单独做：先删除/改写导入方，再移动实现。

## 归档前必须处理的运行时依赖（提案 §2.6）

| 依赖 | 谁在读 | 处理 |
|---|---|---|
| `calibration/reliability_table.json` | `utils/reliability.py`（**默认路径**）、控制器每会话 | 先切新版配置或临时兼容表，再归档旧表；`reliability_table_v1.json` 只给复算 |
| `calibration/set/*`、`sft_data/split_v2.json` | 构建脚本、划分断言 | 新代次另立划分与原始分数目录，旧目录随后归档 |
| `sft_data/FROZEN.json` | 三个写入入口的门禁 | 跟随代码保留；代次化命名后仍以它为准 |
| `experts/trufor.py` | 外部对照（不进主链） | 保留到新专家评测完成 |
| `dataset/` | 数据源 | 继续使用，不归档（提案 §2.3） |

## 归档时**必须**同时给出的东西

1. `PATH_MAP.json`：原路径 → 归档路径，供历史复算解析（提案 §2.5）；
2. 移动**之后**用本清单复跑一次，逐行核对摘要（本文件的 `--verify`）；
3. 代次命名与冻结登记升级（见 `NEW_GENERATION_NAMING.md`），避免冻结身份永久占用新版批次名。
