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
  5. **写入入口清单是七处，不是三处**：收尾复查发现 `admit_trajectories_g4.py`（默认写 `sft_data/admission_report.json`，即**归档路径**）、`build_review_workstation.py`、`record_review.py`（处置台账写在批次的 review 目录下）同样写批次材料却没有 `--generation`。三处一并加上**必填** `--generation` + 冻结拒收，并新增 `test_every_batch_writer_demands_an_explicit_generation`：对七个入口都不带参数运行，断言 argparse 在"必填项"里点名 `--generation`（"必填"而不是"可选"）。清单写进测试，加脚本就得加进去。
* **涉及/修改的文件清单**：`README.md`、`CURRENT_PROGRAM_ARCHITECTURE.md`、`KNOWN_ISSUES.md`、`ARCHIVE_INVENTORY.md`、`NEW_GENERATION_NAMING.md`、`plan.md`、`sft_data/BATCH_INVENTORY.md (Regenerated)`、`scripts/batch_inventory.py (Modified — 导出解析走映射 + 归属措辞 + 台账路径表)`、`scripts/g7_label_transitions.py (Modified — resolve_path + 空读报错 + resolved_from_archive)`、`scripts/admit_trajectories_g4.py`、`scripts/build_review_workstation.py`、`scripts/record_review.py (Modified — 必填 --generation + 冻结拒收)`、`tests/test_batch_binding.py (Modified — 归档感知 + 单文件 resolve 回归 + 九条去跳过 + 七入口必填项测试)`、`tests/test_evidence_consistency.py`、`tests/test_train_lora_g4.py`、`utils/batch.py (Modified — 单文件尾斜杠修复)`、`claude_operation_log.md`
* **执行结果与验证状态**：`bash scripts/run_tests.sh core` **348 通过**；`module` **919 通过**；`full` **1158 通过 / 3 跳过**（`module`/`full` 的失败全部修完，无遗留）。`python scripts/archive_v1.py --verify` 与 `python scripts/archive_inventory.py --verify` 在文档改动后复跑仍通过（归档件未被后续编辑碰到）。**未运行 GPU。**
* **置信度或遗留待办（TODO）**：①三项归档后的假零/静默跳过已修，但同类风险不排除还有——判据是"空读要报错或标注，不能产出看着像结论的零"②README 仍保留 g7 代的数字（§7.3）作为方法层证据，若换专家后结论变化需在阶段 3 复核并更新③阶段 2 起需要用户提供的事实仍未到（Qwen3-VL 版本/推理模式、NPR/UnivFD 权重与许可、noise 保留判据、重复轮处置口径）

### 2026-10-15 - 用户三条指正：漏掉的写入入口、冻结只查到批次、noise 口径不一致

* **当前操作动作**：按用户指正修正三处——①"全部写入入口已完成"的结论不成立，`build_review_from_trajectories.py` 仍用无代次的 `batch_paths`；②冻结检查只查具体批次，不查代次；③新计划把 noise 列为退役，与提案不符。顺带把同一轮全量 sweep 找到的同类缺口一并补上。
* **对应计划锚点**：`plan.md` §1 第 3/3b 项、§3、§4；`NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md` §1
* **核心变更说明**：
  1. **写入入口是九个，不是七个**：用户在 `build_review_from_trajectories.py`（写评审样本 + `review_only.json` + VERSIONS 戳，还能建工作台）之外，全量 sweep 又查出 `g7_select_sources.py`（写变体与 `selection.json`）。两个都补上必填 `--generation` + 冻结拒收；测试里的 `WRITERS` 清单同步扩到九个，并在注释里写明"这份清单是用户找出的缺口促成的"。
  2. **冻结检查升级为两级**（`utils/batch.py::assert_generation_is_writable`）：①**未登记代次**一律拒（`--generation g9` 是拼写错误，不是场景，无 `--allow`）；②**冻结代次下不得新建任何批次**——`--generation g7 --batch b9` 即使加 `--allow-frozen` 也拒，这一条正是 (代次, 批次) 两段身份的用意；③**已登记的冻结批次**仍可按原语义 `--allow-frozen` 做带标记的复算。三种情形都用真 CLI 跑过（`build_final_v2.py`），另加 3 项测试（新批次名/未登记代次/带标记复算）。
  3. **noise 口径统一到提案**：已决定退出的是 **jpeg 与 frequency_v2**；**noise 是候选保留**，与 NPR/UnivFD 同标准评测，缺乏增益与互补性才退出。改了 `plan.md` §3（新增保留表）与 §4、`README.md` §0.2 与 §7.2、`CURRENT_PROGRAM_ARCHITECTURE.md` §4、`KNOWN_ISSUES.md` C3（用户原句把 noise 一并列入退出，已按提案更正），`plan.md` §7 的待提供事实也改成"评测口径与保留判据"。
  4. **同一轮又清掉三处归档后空读**（用户未点名，但属同一判据）：`g7_abstention_diagnosis.py`、`g7_ledger_replay.py`（默认读 g7_b1 轨迹）、`build_g7_group_d.py`（读 237 集的变体清单去做"新鲜组"排除——跳过排除会让"新鲜"名不副实）、`cpu_acceptance_r5.py` 的隔离检查（对不存在的目录求摘要，等于恒真的空检查）、`g7_b2_report.py`（改为标注 `resolved_from_archive`）。
* **涉及/修改的文件清单**：`utils/batch.py (Modified — generation_register/generation_status/assert_generation_is_writable + 两级判定顺序)`、`scripts/build_review_from_trajectories.py`、`scripts/g7_select_sources.py`、`scripts/g7_abstention_diagnosis.py`、`scripts/g7_ledger_replay.py`、`scripts/build_g7_group_d.py`、`scripts/cpu_acceptance_r5.py`、`scripts/g7_b2_report.py (Modified — 必填代次/归档感知)`、`tests/test_batch_binding.py (Modified — WRITERS 扩到九个 + 三项代次级冻结测试)`、`plan.md`、`README.md`、`CURRENT_PROGRAM_ARCHITECTURE.md`、`KNOWN_ISSUES.md`、`claude_operation_log.md`
* **执行结果与验证状态**：`core` **348 通过**、`module` **919 通过**、`full` **1161 通过 / 3 跳过**；真 CLI 验证：`--generation g7 --batch b9` 被拒（提到"不得在其下新建任何批次"）、`--generation g9` 被拒（列出登记表只有 g7/g8）、`--generation g7 --batch g7_b2 --allow-frozen` 放行。**未运行 GPU。**
* **置信度或遗留待办（TODO）**：①"九个"是**当前**的完整清单，不是永久结论——同一轮就补了两次，故把清单放进测试并写明判据（sweep `batch_paths(` 调用方 + 直接写 `sft_data/` 的脚本）②`apply_review.py`/`build_review_workstation.py` 等模块级默认常量（如 `FINAL_V2_DIR`）仍在文件里，但 `main()` 一律用 `batch_paths(batch, generation)` 覆盖；常量是历史默认值，清理由阶段 4 的退役一并做③阶段 2 仍需用户事实（Qwen3-VL 版本/推理模式、NPR/UnivFD 权重与许可、noise 评测口径、重复轮处置口径）

### 2026-10-15 - 收口复查：把"完整性"从口头结论改成可复核的分档

* **当前操作动作**：上一条只按 `batch_paths(` 调用方判完整性，结论被用户指出不成立。改为**两条独立口径**重查：①调用 `batch_paths` 的脚本；②**直接写/读 `sft_data/` 路径**而不经它的脚本。后一条又查出两个只读分析脚本会空读归档。
* **对应计划锚点**：`plan.md` §1 第 3 项（写入入口清单）
* **核心变更说明**：
  1. **`g7_calibration_audit.py`**：读 g7_b1 轨迹做"参考后验是否依赖调用顺序"的审计——归档后会在**零会话里找不到任何顺序依赖**，看起来像结论。改走 `resolve_path`，并在零轨迹时直接报错（点明目录）。
  2. **`screen_visual_grounding.py`**：读 v1 训练集做视觉落地筛查——空筛查会被当成"没有可疑项"。同样改走映射 + 空读报错。
  3. **`g7_acceptance.py` 无需改动**（读校准集与 `split_v2.json`，二者都没移动）——这一条也写进日志，避免下次有人再查一遍。
  4. **分档口径固定下来**：脚本只有三种状态，且状态在文件里可见——**九个写入入口**（必填 `--generation` + 冻结拒收）、**十个只读/复算脚本**（走 `PATH_MAP` 解析，空读报错或标注）、**v1 流水线脚本**（对归档输入直接拒绝）。这比"全部已迁移"这种断言更可复核。
* **涉及/修改的文件清单**：`scripts/g7_calibration_audit.py`、`scripts/screen_visual_grounding.py`、`claude_operation_log.md`
* **执行结果与验证状态**：`module` **919 通过**、`full` **1161 通过 / 3 跳过**；两条口径的审计输出已打印在会话中（9 写入 ✅ / 3 复算 ✅ / 其余逐个判定）。**未运行 GPU。**
* **置信度或遗留待办（TODO）**：①分档仍是"当前文件状态"的结论，不是不变量——判据已写进测试（写入入口）与日志（两条口径的命令）②阶段 2 仍需用户事实（Qwen3-VL 版本/推理模式、NPR/UnivFD 权重与许可、noise 评测口径、重复轮处置口径）

### 2026-10-15 - 根目录文档归类：根目录只留四份，其余进 docs/{design,development,archive}

* **当前操作动作**：按用户七条规则整理根目录文档——九个文件归类移动、提案有效内容整合进 `plan.md`、新旧路径引用全部同步。**不改写历史资料、不改运行逻辑、不删数据。**
* **对应计划锚点**：`plan.md` §1b（本轮新增的计划节，先写计划后动手）
* **核心变更说明**：
  1. **移动九个文件**（全部 `git mv`，`git status` 只见 `R`）：`docs/design/` 收 `active_forensic_agent_tasks.md`、`Reasoning_Framework.md`、`CURRENT_PROGRAM_ARCHITECTURE.md`、`NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md`；`docs/archive/` 收 `ARCHIVE_INVENTORY.md`、`archive_inventory.json`、`NEW_GENERATION_NAMING.md`；`docs/development/` 收 `TESTING.md`、`agent.md`。根目录只剩 `README.md`/`plan.md`/`claude_operation_log.md`/`KNOWN_ISSUES.md`。
  2. **`agent.md` 的位置先核实后移动**：`.claude/settings.local.json` 无 hook、无任何代码 `open()` 它、根目录不存在 Claude Code 会读的 `CLAUDE.md`；代码里的引用都是散文（"agent.md §3.2"）。结论：**没有工具依赖其位置**，因此不保留根目录入口，随本次移入 `docs/development/`，并把散文引用补上目录前缀（10 个文件）。
  3. **提案整合进 plan.md**（避免两份并行维护）：`plan.md` §2–§6 重写为「已确认决策（D1–D10）／八步实施与验收／视觉分析前置验收／证据包与分数边界／通用问题对照」，提案原文顶部加横幅"内容已整合进 plan.md §2–§6，不再是执行依据"，移入 `docs/design/` 留存。
  4. **配套脚本与测试改路径**：`scripts/archive_inventory.py` 的产物改为 `docs/archive/{ARCHIVE_INVENTORY.md,archive_inventory.json}`、`scripts/archive_v1.py` 改读新位置、`tests/test_config_fingerprint.py` 读 `docs/design/CURRENT_PROGRAM_ARCHITECTURE.md` 与 `docs/development/TESTING.md`。
  5. **新增 `tests/test_doc_layout.py`（8 项）**：①根目录 `.md` 集合必须恰好是那四份；②九个归类文件都在预期位置；③**当前文档的相对链接全部可解析**（归档的 v1 文档豁免——它们保持原样）；④提案横幅仍在。链接这一项以前没有任何机器检查，靠人看。
  6. **执行中发现两处"核验失效"并修掉**（都是核验本身不成立）：①`archive_v1.py --verify` 以 `HEAD` 为移动前参照，移动提交后 `HEAD:plan.md` 已是新计划，把归档的 v1 计划误报成"内容变更"——改为以**新增归档路径的提交（即移动提交）的父提交**为参照，比原来"必须提交前跑"更强；②`archive_inventory.py --verify` 不认识文档归类，把移入 `docs/design/` 的提案误报为"缺失"——改为按新位置说明并计数（快照仍保持原样，不重生成）。
* **涉及/修改的文件清单**：移动：`active_forensic_agent_tasks.md`→`docs/design/`、`Reasoning_Framework.md`→`docs/design/`、`CURRENT_PROGRAM_ARCHITECTURE.md`→`docs/design/`、`NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md`→`docs/design/`、`ARCHIVE_INVENTORY.md`→`docs/archive/`、`archive_inventory.json`→`docs/archive/`、`NEW_GENERATION_NAMING.md`→`docs/archive/`、`TESTING.md`→`docs/development/`、`agent.md`→`docs/development/`；修改：`plan.md (Modified — §1b 计划与结果、§2–§8 整合提案、指针路径)`、`README.md (Modified — 文档表/维护表/§0 布局说明/agent.md 路径/badge)`、`KNOWN_ISSUES.md (Modified — 四处文档路径)`、`scripts/archive_inventory.py`、`scripts/archive_v1.py`、`tests/test_config_fingerprint.py`、`tests/test_doc_layout.py (Created)`、`docs/design/NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md (Modified — 状态横幅)`、`docs/design/Reasoning_Framework.md (Modified — 位置说明)`、`docs/archive/ARCHIVE_INVENTORY.md (Modified — 相对链接)`、`docs/development/agent.md (Modified — 位置说明)`、七个代码文件（`agent.md` 散文引用补前缀）
* **执行结果与验证状态**：`git status` 九个 `R` 无内容增删；**链接检查 20 条相对链接、0 失效**（并入 `tests/test_doc_layout.py`）；`core` **348 通过**、`module` **927 通过**、`full` **1169 通过 / 3 跳过**；`python scripts/archive_v1.py --verify`（34 项 / 2141 文件 / 438 页登记改写）与 `python scripts/archive_inventory.py --verify`（移动过的 25 项一致 + 1 项文档归类 + 19 项未移动）在改动后**复跑通过**。**未运行 GPU，未删除任何文件。**
* **置信度或遗留待办（TODO）**：①归档的 v1 文档（`archive/legacy_v1_20261015/**`）保持原样、其内部链接不再维护——`docs/**` 的链接测试只覆盖当前文档，这个边界写在测试文件的注释里②`agent.md` 仍写着"项目根目录维护四个核心控制文件"，已加一句位置说明而未改条文（规范内容不动，等用户决定是否重写）③`docs/design/Reasoning_Framework.md` 与 `active_forensic_agent_tasks.md` 里的 plan 小节号指向 v1 计划，已在 `Reasoning_Framework.md` 顶部注明去归档里找，任务书保持原样
