# 批次清单（自动生成，勿手改）

生成时间：2026-10-13T21:59:13｜命令：`python scripts/batch_inventory.py`

**三类材料不得混用**：历史批次 / 待审核批次 / 合格训练数据。本表由文件本身生成（轨迹、清单、`VERSIONS.json`、审核记录），缺失的版本一律写“未记录”，不猜。

**两条一直容易被忘掉的规则**：

1. **重新渲染工作台 ≠ 重新生成轨迹**——评审样本可以重渲染（例如修好转换器后），轨迹内容不变；`review_only.json` 标记的批次是评审材料，未过准入。
2. **自动检查通过 ≠ 允许训练**——账本、门禁、筛查器的“通过”只说明没有命中已定义的缺陷模式；进入 SFT 需要：人工审核写回处置 + `apply_review` 出集 + `build_final_v2` 门禁全过。

## g7_b1

- 类别：**冻结 · 历史实验（已审核，仍不用于训练）**
- 轨迹：210 条｜策略版本：v2
- 版本（来自 VERSIONS.json）：策略 v2｜提示词 未记录（早于提示词指纹入批规范）｜校准表 d08051253a8278c3（拟合组 未记录）｜代码 未记录
- 样本：清单指纹 bb64cb8ec42788df｜清单样本 145｜评审样本 145｜准入已应用：是
- 审核：本批样本 145 条｜浏览器导出 145 条（sft_data/review/g7_b1/review_decisions_g7_b1.json）｜写回仓库的处置 0 条（该导出尚未 `record_review --import`）｜全局 dispositions 237 行（属于更早的 237 集）

## g7_b2

- 类别：**冻结 · 历史实验（未审核；新材料由新版专家生成）**
- 轨迹：56 条｜策略版本：v3
- 版本（来自 VERSIONS.json）：策略 v3｜提示词 fd8e9307fe85afd4｜校准表 3d16c70538a1d0bc（拟合组 未记录（当时无此字段））｜代码 d2d34e1
- 样本：清单指纹 未记录｜清单样本 未记录｜评审样本 56｜准入已应用：否（评审材料）
- 审核：本批样本 56 条｜浏览器导出 0 条（sft_data/review/review_decisions.json）｜写回仓库的处置 0 条｜全局 dispositions 237 行（属于更早的 237 集）

---

## 路径约定

| 路径 | 内容 | 能否训练 |
|------|------|----------|
| `sft_data/train/final/` | 最早的冻结基线（v1 时代） | 只作回归对照 |
| `sft_data/train/final_v2/` | G4 转换产物（v2 时代，含 237 集） | 只作对照 |
| `sft_data/train/batches/<id>/` | 分批产物；`review_only.json` 表示评审材料 | 仅当该批有审核处置且过门禁 |
| `sft_data/trajectories_<id>/` | 批次轨迹（只读，含 `VERSIONS.json`） | 不直接训练 |
| `sft_data/trajectories_<id>_<原因>/` | **留证归档**：因缺陷作废的批次 | 不得用于任何统计或训练 |
| `sft_data/review/<id>/` | 工作台页面（可重渲染） | — |

进入 SFT 的唯一路径：`record_review --import` → `apply_review` → `build_final_v2`（门禁）。
