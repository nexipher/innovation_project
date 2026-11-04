# GenImage 使用历史审计（2026-11-02）

本文件由 `scripts/audit_genimage_usage.py` 生成；旧划分、轨迹、校准、审核记录只读。
逐源记录和双哈希见 `sft_data/g8_audit/genimage_usage_history_20261102.json`。

## 结论

- 可证明的实际训练只覆盖 **20 个源**（G4 final_v2 的 237 个样本）；旧 train 成员本身不记为训练。
- 新 test 的旧划分来源：train **1258**、val **37**、test **42**、holdout **13**。
- 新 test 中 **138** 个源有已知使用，**1212** 个仅有划分归属、使用未知。
- **没有可靠、可证明未见的 GenImage q95 最终测试**；新 test 只保留为隔离的污染/未知开发检查，不用于专家选择。
- 旧 test 286 源中 **30** 个有先验使用，**256** 个仅划分归属；旧 test 也不是天然干净，继续封存。

## 新划分 × 旧划分

| 新划分 | 输入 | 旧 train | 旧 val | 旧 test | holdout | 无已知使用 | 有已知使用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dev | 6299 | 5827 | 202 | 204 | 66 | 5619 | 680 |
| val | 1350 | 1246 | 45 | 40 | 19 | 1192 | 158 |
| test | 1350 | 1258 | 37 | 42 | 13 | 1212 | 138 |
| excluded_duplicate | 1 | 1 | 0 | 0 | 0 | 1 | 0 |

## 使用类别

| 类别 | 说明 |
| --- | --- |
| `known_train` | 有逐源训练样本和 LoRA run manifest；仅 20 源 |
| `known_sft_data` | 进入过构造的 SFT 数据，但无实际训练证据；不等于已训练 |
| `known_fit` | 98 个 G2 校准 holdout；其中 A 组用于 G7 拟合，B/C 用于选择/验收 |
| `known_dev_eval` | 旧 val 284 源、轨迹/Agent session、G7 验收/D、smoke 等 |
| `known_review` | b1/b2 人工审核与旧 review dispositions |
| `split_only_unknown` | 只有旧/新划分归属，无已知使用；**不等于 clean/unseen** |

## 评测口径

| 集合 | 用途 |
| --- | --- |
| `dev_all`（6299） | 描述性全量对照 |
| `dev_no_known_use`（5619） | 主专家选择集 |
| `val_all`（1350） | 描述性/确认 |
| `val_no_known_use`（1192） | 确认/选择集 |
| `test_sealed`（1350） | 隔离；不是未见测试，不用 |
| `old_test_sealed`（286） | 继续封存；30 源有先验使用 |
| `mnw_q95_existing`（100） | 单独报告；不混入 GenImage、不训练/拟合 |

共享 Real：dev 700 张、其中无已知使用 435 张；val 150 张、其中无已知使用 93 张。同一批 Real 在八个生成器组间复用，组间不独立；不报告跨组合并 AUROC。

## 证据与限制

- 旧 `split_v2`：8902 源（train 8332 / val 284 / test 286），holdout 98。
- GenImage q95 没有既有 PROBE/SPAI 测量；MNW/DIV2K 的 q95 既有测量只用于单独报告，不重跑、不混入。
- 旧 val 284 源全部进入过 `sft_data/g8_eval` 的 val 评测；新 test 中的 37 个old-val 源因此有开发评测历史。
- 训练记录不完整：除 20 个 G4 源外，不能证明其余旧 train 源未用于训练；因此 `split_only_unknown` 不是 clean。
- 旧 test 的 30 个先验使用源也意味着旧 test 不能默认当作未见最终测试。

## 产物

- JSON：`sft_data/g8_audit/genimage_usage_history_20261102.json`
- Dev manifest：`dataset/GenImage_q95/eval_dev_manifest_20261102.json`（6299 输入）
- Val manifest：`dataset/GenImage_q95/eval_val_manifest_20261102.json`（1350 输入）
- 不生成 test manifest；test 保持隔离。
