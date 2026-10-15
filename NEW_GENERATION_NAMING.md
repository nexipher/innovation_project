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

**目录承担代次，批次名保持单纯**（不采用 `g8_b1` 这类把两层含义挤进一个名字的写法）：

```text
sft_data/g8/
├── trajectories/b1/
├── variants/b1/
├── review/b1/
└── train/batches/b1/
```

- 批次名就是 `b1`、`b2`……**不含代次**；代次是路径的一段，由调用方**显式传入**。
- 因此不存在"猜代次"这回事：没有代次参数就没有输出路径，冲突无从产生。
- 旧布局（`sft_data/trajectories_g7_b1/`、`sft_data/train/batches/g7_b1/` 等平铺命名）
  **保持原样并被归档**，只服务于历史复算；新版一律走 `sft_data/<generation>/`。
- 两套布局并存但**不混用**：`generation` 给 `None` ⇒ 旧平铺布局（且只允许历史 id）；
  给了 `generation` ⇒ 新嵌套布局。见 §3.1。

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

`utils/batch.py` 的改动（约四处）：

```python
def batch_paths(batch_id: str, generation: Optional[str] = None) -> Dict[str, str]:
    """generation=None -> 旧平铺布局（仅历史 id）；否则 sft_data/<generation>/<kind>/<batch>/。"""

def assert_not_frozen_batch(batch: str, purpose: str, generation: str,
                            allow: bool = False) -> None:
    """generation 必填、无默认值：不猜，也不从批次名推——名字里已经没有代次了。"""
```

- **显式代次**：三个写入入口（`generate_trajectories_g4` / `build_final_v2` / `apply_review`）
  新增**必填** `--generation`；缺了就报错退出（不允许"默认 g8"）。
- **冻结身份 = (显式代次, 批次)**：`frozen_batches()` 返回 `{(generation, batch): entry}`。
- **历史兼容**：`generation=None` 时走旧平铺布局，且**只允许登记表里已有的历史 id**
  （`g4_trial`、`g7_b1`、`g7_b2`）——它们各自的代次写在登记表里（都是 `g7`）。
  新版脚本不经过这条路径。
- **冲突即报错**：`(generation, batch)` 已在冻结登记中 ⇒ 拒绝；未登记的代次 ⇒ 拒绝；
  同一代次目录内批次名重复 ⇒ 拒绝。

**验收**：`sft_data/g8/...` 下的新批次 `b1` 通过；`g7_b1`/`g7_b2`/`g4_trial` 仍拒绝
（现有 5 项测试保持通过）；**不带 `--generation` 的新版调用直接报错**；
历史 id 不带 generation 仍可解析到旧布局（只读/复算场景）。

### 3.2 代次登记表 `sft_data/GENERATIONS.json`（**不能替代批次身份**）

代次表只回答"这一代大体是什么"；**同一代次里也会换模型版本、换专家权重、改预处理**，
所以**每份 `VERSIONS.json` 仍必须记录该批次实际使用的**模型版本、专家权重与预处理，
不能靠"同代次"推断。`write_batch_versions` 因此要扩展为：

```json
{"generation": "g8", "root": "sft_data/g8", "created_at": "…",
 "model": {"name": "Qwen3-VL-8B", "checkpoint": "<待确认>", "inference_mode": "<待确认>"},
 "experts": ["probe", "noise", "npr", "univfd"],
 "prompt_digest": "…", "calibration": {"digest": "…", "fitted_on": "…"},
 "status": "active"}
```

```json
// VERSIONS.json（每批次，新增字段）
{"generation": "g8",
 "model": {"name": "Qwen3-VL-8B", "checkpoint": "<该批次实际加载的>",
           "inference_mode": "<thinking|non-thinking>", "weights_sha256": "…"},
 "experts": {"probe": {"weights_sha256": "…", "preprocess": "<resize/normalize 说明>"},
             "npr": {...}},
 "prompt_digest": "…", "calibration": {...}, "fingerprint": {...}}
```

提案 §9「配置版本绑定」要求模型、提示词、专家权重、预处理、证据 schema 与代码身份进批次记录。
现状：`VERSIONS.json` 有代码/提示词/专家名/校准，**缺模型身份、权重摘要与预处理**——
这三项按上表补进**每一批次**，代次表只作索引。

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

**新布局不产生新的映射需求**：`sft_data/g8/...` 是全新路径，与归档无关；
`PATH_MAP.json` 只描述"旧平铺路径 → 归档路径"，且只在 `generation=None` 的历史复算里被查询。

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
| `calibration/reliability_table.json` | `utils/reliability.py` 默认路径；控制器每会话 | **不建新版运行时校准表**（阶段 4 已决定：新版运行时关闭参考后验与预计收益）。正确顺序是**先解除运行依赖**——停算后验/效用、不再查表、方向检查改按经核实的原始 score 语义；解除之后这张表对运行时**没有读者**，旧表归档即可。临时兼容表 + Mock 只供迁移冒烟，**不构成新版正式前置**；离线校准对照（C5）是后续独立课题 |
| `calibration/set/*`（划分、原始分数） | 构建脚本 | 新代次建立自己的 set 目录；旧目录随归档移动并按 PATH_MAP 解析 |
| `sft_data/split_v2.json` | 运行与构建都读 | 新代次沿用**同一划分**（数据源不变），但最终测试源需按 KNOWN_ISSUES「数据泄漏」另行划定并记录使用史 |
| `sft_data/FROZEN.json` | 三个写入入口的门禁 | 跟随代码保留（不归档） |
| `experts/trufor.py` | 外部对照，不进主链 | 保留到新专家评测完成 |
| `dataset/` | 数据源 | 继续使用，不归档 |

**临时兼容配置的红线**：只用于"模型迁移冒烟不依赖已封存路径"，**不得用于新版正式生成**
（提案 §2.6）。实现上给 `VERSIONS.json` 加 `"compatibility_mode": true` 并让它进台账类别，
一眼能看出哪些产物是兼容模式跑出来的。

**顺序（不变）**：解除运行依赖 → 冒烟（可用兼容/Mock）→ 归档旧表 → 阶段 4 彻底关闭旧数值路径。
"先建一张新版校准表"不在路线里。

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
   **代码文件一个都不动**（`role=code`，见下"归档 ≠ 退役"）；
2. 写 `MANIFEST.json`（提案 §2 的四段结构）与 `PATH_MAP.json`；
3. **移动前**对待归档目录做一次全量摘要（清单已对 docs/history 全量；大目录的部分摘要
   不证明完整备份），**移动后**用 `python scripts/archive_inventory.py --verify` 逐行核对；
4. 建新版 `plan.md`、`claude_operation_log.md`（同名，顶部注明历史入口）；
   **并按提案 §2 的批注统一更新一次 README、删除旧字段**——即把 README 里
   描述 v1 代次的语句收敛成"指向归档"的一句话，只保留当前能运行的内容；
5. 改 `FROZEN.json`（加 `generation` 字段）与 `utils/batch.py`（`batch_paths` 支持嵌套布局、
   `assert_not_frozen_batch` 要求显式代次）、加 `GENERATIONS.json`（§3.2）、加 `resolve_path`（§3.3）；
   三个写入脚本加**必填** `--generation`；建 `sft_data/g8/` 空骨架（trajectories/variants/review/train/batches）；
6. 跑 `scripts/run_tests.sh core`，确认门禁与路径相关测试全绿（预计需要改 3–5 处测试里的硬编码路径）。

**归档体量（按新口径）**：待归档 = docs+history 共 30 项、**约 119 MB**（已去父子重叠）；
`runtime`/`code`/`external`/`dataset` 共 15 项、约 6.35 GB **保持原位**。
两个数字都不是"磁盘总占用"，日志里曾经的 6.32 GB 也不能当归档量。

### 归档 ≠ 退役（用户点名）

`role=code` 的项是**仍被导入的已退役实现**。以 `experts/frequency.py` 为例，它仍被
`experts/__init__.py`（包导出）、`scripts/calibrate_experts.py`、`scripts/evaluate_experts_g2.py`、
`tests/conftest.py`、`tests/test_frequency_expert.py`、`tests/test_expert_admission.py` 导入——
**移动它会直接破坏包导入与测试**。因此：

- 阶段 1 **只归档数据**，代码原地不动；
- 代码退役放到阶段 4（专家退出）与导入方一起改：先改导入/导出与专属测试，再移动实现；
- 清单对每个 `code` 项自动列出导入方（见 `ARCHIVE_INVENTORY.md` 的说明列），移动前必须为空。

不在此清单内：Qwen3-VL 权重下载、专家权重下载、任何 GPU 运行。
