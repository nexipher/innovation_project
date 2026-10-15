# 操作日志（当前代次 g8）

> **本文件从 2026-10-15 起只记当前代次的操作，只追加。**
> v1 操作日志（2026-07-17 ~ 2026-10-15，共 6 个阶段）已归档：
> `archive/legacy_v1_20261015/docs/claude_operation_log_v1.md`；
> 同期计划：`archive/legacy_v1_20261015/docs/plan_v1.md`。
>
> 每条六字段：当前操作动作｜对应计划锚点｜核心变更说明｜涉及/修改的文件清单｜
> 执行结果与验证状态｜置信度或遗留待办（TODO）。

---

### 2026-10-15 - 阶段 1：代次布局、冻结升级与 v1 归档移动

* **当前操作动作**：执行阶段 1 的实施与**归档移动本身**——把 docs+history 共 34 项移入 `archive/legacy_v1_20261015/`，写清单与路径映射，逐文件核对，并建立新版 `plan.md` / `claude_operation_log.md`。**代码文件一个都没移动，未删除任何文件，未解除任何冻结门禁。**
* **对应计划锚点**：`NEW_GENERATION_NAMING.md` §6 动作清单 1–4（用户已确认"可以进入第1阶段实现"）；`NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md` §2
* **核心变更说明**：
  1. **归档用脚本完成，不用手敲 `mv`**：`scripts/archive_v1.py`（`--dry-run` 先看计划，`--record-links` 登记改写，`--verify` 逐文件核对）。移动前逐项记 文件数/字节数/内容摘要，写入 `MANIFEST.json`（四段结构 + `stays_in_place` + `limits`）与 `PATH_MAP.json`（旧路径 → 归档路径）。**没移动**：`sft_data/review/dispositions.jsonl`（`record_review` 运行时读写、增量追加，搬走即中断工具）、`FROZEN.json`/`GENERATIONS.json`/`split_v2.json`（跟代码）、`calibration/`（校准表仍是运行时默认路径，须先解除依赖再归，属阶段 4）、`dataset/`。
  2. **归档页面的图片链接按映射改写，且逐页登记**：页面里三类链接深度不同（`dataset/`、`traces/`、`variants_*`），做法是"按**原**页面位置解析 → 走映射 → 按**新**位置重算相对路径"，而不是把 `../` 数量写死。438 页被改写，每页的**改写前摘要与字节数**记进 `MANIFEST.json` 的 `links_rewritten`；`record_links()` 会把变换重新施加到 `HEAD` 的 blob 上比对——对不上就报错退出，**"改写之外还改了什么"因此不可能被瞒过去**。
  3. **两道核对，各自说清覆盖了什么**：`archive_inventory.py --verify`——移动过的 27 项与移动前清单逐项一致，2 项（`plan.md`、本日志）归档副本与 `HEAD` 一致、差异只是清单快照早于其后一次正当修改，18 项从未移动（快照不构成一致性判据，明说未检）；`archive_v1.py --verify`——34 项、2141 个文件逐文件比对 `HEAD` blob，216 个 git 未收录的二进制（变体 PNG）按文件数与字节数核对（≈98 MB，同盘 `rename`，未逐块校验）。
  4. **核对过程抓到一个真 bug**：`utils/batch.resolve_path()` 对**单文件**条目把 `os.path.join(..., "")` 的尾斜杠带进了 `os.path.exists()`——Linux 下带尾斜杠的文件路径判 False，于是所有单文件映射都误报"不是归档来源"（目录条目不受影响，所以之前没暴露）。改为 `os.path.normpath` 后 27 项才全部核对成功。若不做逐行核对，这个 bug 会一直躺着。
  5. **历史命令不会"假装还能跑"**：`g7_audit_traces.py` 走 `resolve_path` 并在报告里写 `resolved_from_archive`；`batch_inventory.py` 对每批显示"已归档 → 数字读自归档副本"而不是退化成"未记录"；`finalize_sft_data.py`、`train_lora_g4.py` **拒绝**在已归档路径上重建或训练（`train_lora_g4` 要 `--allow-archived` 才放行，且打印"本次训练读的是已归档 v1 数据"横幅）；`build_variants_g4.py` 补上必填 `--generation`（它同样写批次目录）。
  6. **gitignore 跟进**：变体 PNG 原本按 `sft_data/variants*/*.png` 不入库，归档后路径变了，补 `archive/**/variants*/*.{png,jpg}`，否则一提交就会把上百 MB 图像塞进仓库。
* **涉及/修改的文件清单**：`scripts/archive_v1.py (Created)`、`archive/legacy_v1_20261015/{MANIFEST.json,NOTE.md}` + `archive/PATH_MAP.json (Created)`、`archive/legacy_v1_20261015/{docs,experiments,review}/** (Moved — 34 项)`、`scripts/archive_inventory.py (Modified — verify 走 PATH_MAP、跳过含登记改写的目录、与 HEAD 比对判定"快照过期"、报告四类计数)`、`scripts/g7_audit_traces.py (Modified — resolve_path + resolved_from_archive)`、`scripts/batch_inventory.py (Modified — 归档感知 + 报告行 + 路径约定表)`、`scripts/finalize_sft_data.py (Modified — assert_not_archived)`、`scripts/train_lora_g4.py (Modified — assert_not_archived + --allow-archived)`、`scripts/build_variants_g4.py (Modified — 必填 --generation + 冻结拒收)`、`utils/batch.py (Modified — resolve_path 单文件尾斜杠修复)`、`.gitignore (Modified — 归档变体图像不入库)`、`plan.md`、`claude_operation_log.md`（新版建立）
* **执行结果与验证状态**：`python scripts/archive_v1.py --verify` 通过（34 项 / 2141 文件 / 438 页登记改写 / 216 个未入库二进制按计数+体量核对）；`python scripts/archive_inventory.py --verify` 通过（27 项一致 + 2 项与 HEAD 一致 + 18 项未移动未检，全部计数列出）；`core` 层 347 通过、1 失败——即本文件与新版 `plan.md` 建成前的"当前文档"断言（`test_no_current_doc_claims_the_old_default`），写完本文件后复跑通过（见下一条）。**未运行 GPU，未下载任何权重。**
* **置信度或遗留待办（TODO）**：①本阶段剩余：README 统一改写（删旧字段）②归档体量的强/弱证据分清了：git 收录的文件逐文件比对 blob（强），变体 PNG 只有计数+体量（弱，同盘 rename 未逐块校验），已如实写在 MANIFEST 与核对输出里③`calibration/` 的归档要等阶段 4 解除运行时依赖（校准表默认路径）④仍缺用户提供的事实：Qwen3-VL 版本/推理模式、NPR/UnivFD 权重与许可、noise 保留判据、重复轮处置口径（掩码 vs 整条拒收）

### 2026-10-15 - 阶段 1 收尾：文档收敛（删旧字段）与"归档不得让检查静默退化"

* **当前操作动作**：统一改写 README（按提案 §2 的批注"删除旧字段"，只留当前能跑的内容 + 指向归档的一句话），同步 `CURRENT_PROGRAM_ARCHITECTURE.md` 的数据布局段、`KNOWN_ISSUES.md`、`ARCHIVE_INVENTORY.md`、`NEW_GENERATION_NAMING.md` 的状态标注，重建 `sft_data/BATCH_INVENTORY.md`，并修完全部因移动而失效的测试路径。**未改任何代次/冻结语义，未删除数据。**
* **对应计划锚点**：`plan.md` §1 阶段 1 第 9–10 项；`NEW_GENERATION_NAMING.md` §6 第 4 项
* **核心变更说明**：
  1. **README 收敛**：§0 改写成"当前代次 g8 + 布局 + 归档"三段（含三个写入入口**必填** `--generation` 的说明、归档三条使用规矩）；§1/§3/§5 里描述 v1 代次的语句改为一句归属说明；§3.6b 的命令补上 `--generation/--batch`（否则照抄就报错）；§3.8 改为三层测试运行器；§4 目录树换成"新 `sft_data/<gen>` + `archive/`"并注明已归档的旧路径；§7.1/§7.2 由 4.4 KB 的阶段表收敛为"指向归档 + 当前路线图"；§7.3 的实测局限保留（仍是有效的方法层结论），顶部加一句来源标注；§8 补 8.6 数据与归档规范。
  2. **归档之后，三类"静默退化"必须显式**：①`batch_inventory.py` 对 g7_b1 又报"浏览器导出 0 条（无导出文件）"（145 条导出随归档搬走）——改为按 PATH_MAP 解析候选路径并在行内标注"读自归档副本"；②g7_b2 行把仓库根的 237 集导出当成本批文件显示"0 条"，改为"无导出文件（仓库根另有 …，属于更早的批次）"；③`test_batch_binding.py` 里 9 条"检查真实批次"的测试因路径不存在而**静默跳过**（覆盖率悄悄下降）——加 `_batch_samples()`/`_batch_path()` 走 `resolve_path`，跳过只剩 1 条（该批次本就没有 `report_*.json`，归档前也跳过）。
  3. **归档抓到的第二个真 bug**：`g7_label_transitions.py`、`g7_audit_traces.py` 空读归档前的目录会产出**全零矩阵/空报告**（看着像结论）。改为 `resolve_path` + 报告里写 `resolved_from_archive`，并把"没有轨迹记录直接报错"补上。
  4. **回归测试**：为 `resolve_path` 的单文件条目加测试（`test_resolve_path_handles_a_single_moved_file`），复现"join 出尾斜杠 → exists 为假"的原始错误条件。
* **涉及/修改的文件清单**：`README.md`、`CURRENT_PROGRAM_ARCHITECTURE.md`、`KNOWN_ISSUES.md`、`ARCHIVE_INVENTORY.md`、`NEW_GENERATION_NAMING.md`、`plan.md`、`sft_data/BATCH_INVENTORY.md (Regenerated)`、`scripts/batch_inventory.py (Modified — 导出解析走映射 + 归属措辞 + 台账路径表)`、`scripts/g7_label_transitions.py (Modified — resolve_path + 空读报错 + resolved_from_archive)`、`tests/test_batch_binding.py (Modified — 归档感知 + 单文件 resolve 回归 4 处 + 9 条去跳过)`、`tests/test_evidence_consistency.py`、`tests/test_train_lora_g4.py`、`utils/batch.py (Modified — 单文件尾斜杠修复)`、`claude_operation_log.md`
* **执行结果与验证状态**：`bash scripts/run_tests.sh core` **348 通过**；`module` **919 通过**；`full` **1157 通过 / 3 跳过**（`module`/`full` 的失败全部修完，无遗留）。`python scripts/archive_v1.py --verify` 与 `python scripts/archive_inventory.py --verify` 在文档改动后复跑仍通过（归档件未被后续编辑碰到）。**未运行 GPU。**
* **置信度或遗留待办（TODO）**：①三项归档后的假零/静默跳过已修，但同类风险不排除还有——判据是"空读要报错或标注，不能产出看着像结论的零"②README 仍保留 g7 代的数字（§7.3）作为方法层证据，若换专家后结论变化需在阶段 3 复核并更新③阶段 2 起需要用户提供的事实仍未到（Qwen3-VL 版本/推理模式、NPR/UnivFD 权重与许可、noise 保留判据、重复轮处置口径）
