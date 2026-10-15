# 代次命名、冻结登记升级与归档路径映射（方案草案，供确认）

> 配套 `NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md` 的实施细节。**本文件只写方案，不改代码、不移动文件。**
> 生成归档清单的命令：`python scripts/archive_inventory.py`（只读；产物 `ARCHIVE_INVENTORY.md` + `archive_inventory.json`）。
> 最后核对：2026-10-15。

## 1. 为什么需要代次（原因先写清楚）

提案 §2.4：「冻结身份不能只依赖裸批次名，避免永久占用新版命名空间」。
现在的冻结登记是裸名字（`g7_b1`、`g7_b2`、`g4_trial`，见 `sft_data/FROZEN.json`）——
它挡得住复用**同名**批次，但挡不住"新版第一批也叫 b1"。所以身份必须是
**(代次, 批次)** 两段。

## 2. 代次定义

| 代次 | 内容 | 状态 |
|------|------|------|
| `g7`（含更早的 `g4_trial`/237 集） | Qwen2.5-VL + PROBE/noise/jpeg/frequency_v2（v2/v3 策略时代） | **冻结**（`FROZEN.json` 已登记，代码层拒收） |
| `g8` | **本次新代次**：Qwen3-VL-8B（待定版本）+ PROBE / noise / NPR / UnivFD（候选），v3 原则、无参考后验 | 待建 |

命名规则（新版批次一律带代次前缀）：

```text
sft_data/trajectories_g8_b1/        sft_data/variants_g8_b1/
sft_data/train/batches/g8_b1/       sft_data/review/g8_b1/
sft_data/admission_report_g8_b1.json
```

旧目录（`trajectories_g7_b1` 等）**保持原名**——它们将被归档，改名只会让历史引用更难解析。

## 3. 三处代码改动（实施阶段 1 执行，改动很小）

### 3.1 `sft_data/FROZEN.json` → 带代次

```json
{
  "generations": [
    {"generation": "g7", "status": "frozen", "since": "2026-10-15",
     "reason": "Qwen2.5-VL + 旧专家组；换代次后不再生成"},
    {"generation": "g8", "status": "active", "since": null}
  ],
  "frozen": [
    {"generation": "g7", "batch": "g4_trial", "reason": "…"},
    {"generation": "g7", "batch": "g7_b1", "reason": "…"},
    {"generation": "g7", "batch": "g7_b2", "reason": "…"}
  ]
}
```

`utils/batch.py` 的三处小改：
- `frozen_batches()` 返回 `{(generation, batch): entry}`；
- `assert_not_frozen_batch(batch_id, purpose, allow=False)` 增加 `generation="g8"` 默认参数，
  先按 `(generation, batch)` 命中，再回退到裸名匹配（旧的 g7 调用不变）；
- 新增 `generation_of(batch_id)`：`g8_b1 → g8`，`g7_b1 → g7`，`g4_trial → g7`（历史批次映射）。

**验收**：新批次 `g8_b1` 通过；`g7_b1`/`g7_b2`/`g4_trial` 仍拒绝（现有 5 项测试保持通过）。

### 3.2 代次登记表 `sft_data/GENERATIONS.json`

每代次一行，记录身份；新版批次生成时写入（`write_batch_versions` 顺带更新）：

```json
{"generation": "g8", "created_at": "…",
 "model": {"name": "Qwen3-VL-8B", "checkpoint": "<待确认>", "inference_mode": "<待确认>"},
 "experts": ["probe", "noise", "nPR", "univfd"],
 "prompt_digest": "…", "calibration": {"digest": "…", "fitted_on": "…"},
 "status": "active"}
```

提案 §9「配置版本绑定」要求模型、提示词、专家权重、预处理、证据 schema 与代码身份进批次记录
——现在批次有 `VERSIONS.json`（代码/提示词/专家/校准），**代次是它缺的那一层**：模型身份。

### 3.3 归档路径映射 `PATH_MAP.json`

归档移动会打断两类引用：轨迹记录里的 `trace_path`、`VERSIONS.json` 里的路径。
规则（提案 §2.5）：**给映射，不假装旧命令还能直接跑**。

```json
{"archived_at": "…", "archive_root": "archive/legacy_v1_20261015",
 "moved": [
   {"from": "sft_data/trajectories_g7_b2", "to": "archive/legacy_v1_20261015/experiments/trajectories_g7_b2",
    "kind": "dir", "files": 61, "digest": "4aa50568ff820f03", "digest_covers_content": true}
 ]}
```

配套一个只读解析器（`utils/batch.py`）：

```python
def resolve_path(path: str) -> tuple[str, bool]:
    """(存在的路径, 是否来自归档)。找不到时查 PATH_MAP.json，命中即返回归档路径。"""
```

调用方（复算脚本、审计脚本）拿到 `resolved_from_archive=True` 时**必须写进报告**，
这样"这份复算是基于归档副本"是事实的一部分，而不是隐含前提。

## 4. 归档前必须先切的运行时依赖（提案 §2.6）

`ARCHIVE_INVENTORY.md` 里 `role=runtime` 的项不能盲目搬走。切换顺序：

| 依赖 | 现在谁在读 | 切换动作 |
|------|-----------|----------|
| `calibration/reliability_table.json` | `utils/reliability.py` 默认路径；控制器每会话 | 新代次另行生成 `calibration/g8/reliability_table.json`（或先放**临时兼容表 + Mock 专家**），改默认路径后再归档旧表 |
| `calibration/set/*`（划分、原始分数） | 构建脚本 | 新代次建立自己的 set 目录；旧目录随归档移动并按 PATH_MAP 解析 |
| `sft_data/split_v2.json` | 运行与构建都读 | 新代次沿用**同一划分**（数据源不变），但最终测试源需按 KNOWN_ISSUES「数据泄漏」另行划定并记录使用史 |
| `sft_data/FROZEN.json` | 三个写入入口的门禁 | 跟随代码保留（不归档） |
| `experts/trufor.py` | 外部对照，不进主链 | 保留到新专家评测完成 |
| `dataset/` | 数据源 | 继续使用，不归档 |

**临时兼容配置的红线**：只用于"模型迁移冒烟不依赖已封存路径"，**不得用于新版正式生成**
（提案 §2.6）。实现上给 `VERSIONS.json` 加 `"compatibility_mode": true` 并让它进台账类别，
一眼能看出哪些产物是兼容模式跑出来的。

## 5. 新版最小冒烟（阶段 2 的入口，现未实现）

提案 §3 的五项检查需要一个不依赖旧实验路径的入口。建议 `scripts/g8_smoke.py`：

1. 只注册**新版工具箱**（初期：一个真专家 + 一个 Mock 专家返回固定 token）；
2. 用 `MockMLLMClient` 跑通协议（CPU），再用 Qwen3-VL 跑同样序列（GPU，另行授权）；
3. 逐项断言：图像真的进入多模态消息（不是只给路径）／一次合法调用被执行／结果进入正确的后续消息且被准确引用／结案可解析／
   **模型原始回复、最终采用回复、外部控制原因三者分别留存**（提案 §3.5）；
4. 产物写进 `sft_data/g8_smoke/`，不进任何批次目录，不触发冻结门禁。

现有的 `--dry-run` 机制（`MockMLLMClient` + Mock 专家）可直接复用；要新增的只是
"工具箱里只有新专家"的注册路径。

## 6. 实施阶段 1 的确切动作清单（待你确认后执行）

1. `git mv` 或 `mv` 清单中的 `docs`/`history` 项 → `archive/legacy_v1_20261015/{docs,experiments,review}/`；
2. 写 `MANIFEST.json`（提案 §2 的四段结构）与 `PATH_MAP.json`；
3. **移动后用** `python scripts/archive_inventory.py --verify` 逐行核对摘要；
4. 建新版 `plan.md`、`claude_operation_log.md`（同名，顶部注明历史入口）；
   **并按提案 §2 的批注统一更新一次 README、删除旧字段**——即把 README 里
   描述 v1 代次的语句收敛成"指向归档"的一句话，只保留当前能运行的内容；
5. 改 `FROZEN.json` + `utils/batch.py` 三处（§3.1）、加 `GENERATIONS.json`（§3.2）、加 `resolve_path`（§3.3）；
6. 跑 `scripts/run_tests.sh core`，确认门禁与路径相关测试全绿（预计需要改 3–5 处测试里的硬编码路径）。

不在此清单内：Qwen3-VL 权重下载、专家权重下载、任何 GPU 运行。
