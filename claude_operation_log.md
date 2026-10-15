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

### 2026-10-15 - 补一道"不许悄悄取消核对"的闸：归档清单不再被默认覆盖

* **当前操作动作**：文档归类后 `archive_inventory.py` 的产物进了 `docs/archive/`——那正是 `--verify` 的比对基准（归档**之前**的状态）。它的默认模式会直接重写该文件，等于把"东西原来在哪"换成"现在在哪"，核对就悄悄失效了。加显式 `--refresh` 才允许覆盖。
* **对应计划锚点**：`plan.md` §1b 验收第③项（脚本改到新路径）的延伸
* **核心变更说明**：`main()` 在写文件前检查目标是否存在，存在且未给 `--refresh` 就**拒绝并说明理由**（为什么这个文件不能随手重拍）；新增测试用子进程跑一次生成器，断言①退出码非零、②stderr 提到 `--refresh`、③**清单文件的摘要前后一致**（被拒绝时确实一个字节都没写）。
* **涉及/修改的文件清单**：`scripts/archive_inventory.py`、`tests/test_doc_layout.py (Extended — 1 项)`、`README.md (badge)`
* **执行结果与验证状态**：`core` **348 通过**、`module` **928 通过**、`full` **1170 通过 / 3 跳过**；`--verify` 复跑仍通过（25 项一致 + 1 项文档归类 + 19 项未移动）。**未重写快照。**
* **置信度或遗留待办（TODO）**：①下一次真正的归档才需要 `--refresh`（届时新快照会覆盖旧的，属预期）②同类"默认就覆盖证据"的脚本若还有，判据是"产物是不是核对基准"

### 2026-10-15 - 阶段 2 开工：模型固定 revision、权重下载、Qwen3 客户端与最小冒烟（CPU 部分）

* **当前操作动作**：按用户指令推进阶段 2 的前三件事——①固定模型与 revision ②核实依赖/磁盘/权重目录并准备下载 ③实现 Qwen3 客户端与最小冒烟入口，先跑 CPU 协议测试。
* **对应计划锚点**：`plan.md` §3b（本轮先写的执行细则）、§3 步骤 2
* **核心变更说明**：
  1. **模型与 revision 固定**：`Qwen/Qwen3-VL-8B-Instruct` @ **`0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`**（2025-10-15）；许可 **apache-2.0**（模型卡明示）；16 个文件、17.55 GB（4 个 safetensors 分片）。已核实镜像可达。
  2. **依赖核实（先查再下）**：本机 `transformers 5.14.0` **已含** `Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor`（无需升级）；`torch 2.5.1+cu124`；无 `flash_attn`（走 sdpa/eager）；`huggingface_hub 1.23.0`。
  3. **磁盘与目录**：`autodl-tmp` 可用 17.50 GB < 模型 17.55 GB（**差 50 MB，放不下**）；`/` 12.42 GB。用户批准后**只删**中断残留 `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-7B-Instruct`（8.1 GB、3 个 `.incomplete`、5 分片只到 2 个）——删除前展示了绝对路径与分片证据，并确认项目实际加载的是另一份完整 ModelScope 快照（`config.QWEN_MODEL_PATH`，5 分片 16.58 GB、0 不完整）；dinov2 缓存（PROBE 主干）与旧模型**未动**。删除后 `/` 可用 21.08 GB。
  4. **下载**：只进 HF 缓存（**不设 `--local-dir`**，避免缓存与本地目录各存一份），带**空间看门狗**（低于 1.5 GB 即停）。第一次失败于 **Xet 传输**（镜像不代理 `cas-server.xethub.hf.co`，权重分片 401，只有小文件下来）→ 加 `HF_HUB_DISABLE_XET=1` 重跑。日志 `/root/autodl-tmp/qwen3vl_download.log`。
  5. **客户端** `mllm/qwen3_client.py`：`Qwen3VLClient` 继承既有 `QwenVLClient`，复用协议实现（消息构建、格式纠错环、`ask()`），只换模型类/处理器；revision **写进代码**；`inference_mode` 必须**验证模板真有 `enable_thinking` 开关**才设置，否则报错（不记录没真正设上的模式）；权重身份（snapshot/分片/字节/index 摘要）load 时免费记录，全量 `weights_sha256` 按需算；新增 `last_input_stats`（输入 token 数、**图像 token 数**、附着的 PIL 图像数）——"图真的进了模型"由此可测。`mllm/__init__.py` 懒导出同步（不破坏 CPU 路径的导入基线）。
  6. **冒烟入口** `scripts/g8_smoke.py`：只注册新版工具箱（CPU：MockProbe + 固定值 noise 桩；GPU：真 PROBE + 固定值桩），产物写 `sft_data/g8_smoke/<时间戳>/`（**不是批次目录**，不触发冻结门禁）。五项断言 + GPU 授权闸门（无 `FORENSIC_ALLOW_GPU=1` 直接拒，避免未经授权就加载 17.5 GB 权重）。
  7. **自查抓到自己一个错**：检查 4 原本在 `result["conversation"]`（字符串转录）里找图像部件——那里按构造永远没有；改为查真正的构造者 `build_messages`（收尾轮历史）+ 真模型时读收尾调用的 `last_input_stats`。
* **涉及/修改的文件清单**：`plan.md (Modified — §3b)`、`mllm/qwen3_client.py (Created)`、`mllm/__init__.py (Modified — 懒导出)`、`scripts/g8_smoke.py (Created)`、`tests/test_g8_smoke.py (Created — 9 项，core)`、`sft_data/g8_smoke/20261015_111456/ (Created — CPU 冒烟产物)`、`claude_operation_log.md`；数据侧：删除中断缓存一项（用户批准），`sft_data/GENERATIONS.json` 待随模型身份一并更新
* **执行结果与验证状态**：**CPU 协议测试五项全过**（图像以 PIL 进入消息、2 次合法调用 probe→noise、后续轮准确引用 0.4242、结案可解析且收尾轮仍带图像、三类记录分别留存）；`core` **357 通过**、`module` **937 通过**；下载进行中（11:07 起，已落地约 4 GB，`/` 余 17 GB，看门狗在跑）。**尚未加载任何权重、未做 GPU 推理。**
* **置信度或遗留待办（TODO）**：①权重下完后才做 GPU 五项验证与显存实测（授权已获，入口 `FORENSIC_ALLOW_GPU=1 python scripts/g8_smoke.py --client qwen3`）②`inference_mode` 的模板开关要在真模板上验证一次（代码里写了断言，尚未在真实模板上跑过）③`SFT_DATA/GENERATIONS.json` 与 README 的模型表要随 revision 一并更新④用户随后会删除 2.5-VL 旧模型腾空间给专家组——我不代删

### 2026-10-15 - 阶段 2 完成：Qwen3-VL 冒烟六项全过（真模型 + 真 PROBE，含实测显存）

* **当前操作动作**：权重下载完成后，按用户四项指令收尾——①依赖/磁盘/目录核实（含清理授权过的中断缓存）②下载并固定 revision ③CPU 冒烟与协议测试 ④GPU 授权后验证图像输入、具体视觉描述、一次工具调用、结果引用、自主结案，并实测显存。
* **对应计划锚点**：`plan.md` §3b（执行细则 + 结果表）
* **核心变更说明**：
  1. **下载**：`Qwen/Qwen3-VL-8B-Instruct`@`0c351dd0…` 17.55 GB 全部落地（exit 0、**0 个不完整分片**），只进 HF 缓存、无 `--local-dir`（不产生第二份）。第一次失败于 **Xet**（镜像不代理 `cas-server.xethub.hf.co`，权重分片 401）→ `HF_HUB_DISABLE_XET=1` 后成功；看门狗全程在跑，`/` 最终余 3.3 GB。
  2. **CPU 预验证（省下 GPU 时间的部分）**：真实 processor 走通"图 → token"——2602 输入 token 中 1024 个图像占位 token，`pixel_values` [4096,1536]，`config.json` 与 tokenizer 两处 `image_token_id` 一致。另由产物确认 **Instruct 版模板无 `enable_thinking`**（⇒ non-thinking，不是选择而是事实）。
  3. **GPU 冒烟（真 Qwen3-VL + 真 PROBE）六项全过**：图像进输入（3817/1024 图像 token）、一次合法调用（模型自主只调 probe，并在 reasoning 里写明"probe 已足够决定性，无需额外调用"）、后续轮引用证据 `aigi_discriminator_score`、结案可解析且收尾轮仍有图像 token、三类记录分别留存、首轮视觉观察 125 字并另列疑似异常。**实测**：权重显存 17.53 GB、整轮峰值 18.6 GB、加载 7.2 s、会话 53.4 s。单例判定正确（Midjourney → Fake，confidence 0.984）。
  4. **三处检查自身的缺陷（都是我的检查错，不是系统错）**：①检查 3 只认桩专家的固定值 → 模型合法地只调 probe 就"失败"；②字段名写成 `raw_value`，实际是 `raw_metric`/`strength`；③真模型引用的是**证据名/工具名**而非小数。最终判据＝"后续轮引用该工具的证据 id、名称、工具名或数值任一"，并报出匹配到的是哪一个。另修：检查 4 一度在字符串转录里找图像部件（按构造永远没有），改为问消息构造者＋读真模型的收尾调用输入统计。
  5. **会话记录归位**：冒烟原先经 `SessionLogger` 把会话写进 `traces/sft_sessions/`（与真实会话混住，且测试每跑一次就多几份文件）——改为 `SessionLogger(sft_dir=<冒烟目录>)`，完整记录（含证据链）与报告同处；手工 dump 的 `session_trace.json` 不再作为主记录（它没有证据链）。已把本次 GPU 的 6 份完整记录移入 `sft_data/g8_smoke/gpu_sessions_20261015/`，删掉测试产生的 mock 残留。
* **涉及/修改的文件清单**：`mllm/qwen3_client.py (Created/Modified — dtype=、采样参数、last_input_stats)`、`scripts/g8_smoke.py (Created/Modified — 六项检查 + processor 预验证 + 授权闸门 + 会话记录定向)`、`tests/test_g8_smoke.py (Created — 11 项，core)`、`mllm/__init__.py (Modified — 懒导出)`、`plan.md (Modified — §3b 结果表)`、`README.md (Modified — §3.7b 冒烟用法 + 模型行)`、`sft_data/GENERATIONS.json (Modified — revision/许可/模式)`、`sft_data/g8_smoke/** (Created — 报告与 GPU 会话记录)`；数据侧：删除中断缓存一项（用户批准）、下载 17.55 GB。
* **执行结果与验证状态**：CPU：`core` 358、`module` 938 通过；GPU：六项全过（报告 `sft_data/g8_smoke/20261015_121549/smoke_report.json`）。**未做**：多会话批量、思考模式（模板没有该开关）、prompt 大规模对比。
* **置信度或遗留待办（TODO）**：①单例通过 ≠ 泛化；下一批要按 §3 步骤 5 做小批端到端（含人工查看原始图像与投递证据包）②v3 目前**仍计算参考后验与预计收益**（报告里能看到 `posterior`、`tool_utilities`）——阶段 4 的解耦尚未做③`inference_mode` 的 thinking 分支在真模板上会正确报错（无开关），该分支本身仍未被真实模板覆盖④噪声桩只是工具链验证，noise/NPR/UnivFD 的真实评测属阶段 3⑤用户将删除 2.5-VL 旧模型腾空间，未代删

### 2026-10-15 - 阶段 3 开工：官方事实核实、分数语义登记、统一评测入口（NPR A/B 全量 val 对比在跑）

* **当前操作动作**：按用户四条口径启动阶段三——①主动核实 NPR/UnivFD 的官方事实（不等用户给）②建分数语义登记表（阶段三与阶段四共用）③建 g8 统一离线评测入口 ④按"CPU 接口 → 单图 → 小批评测"推进，先跑 CPU 可做的部分。
* **对应计划锚点**：`plan.md` §3c（本轮先写的计划）
* **核心变更说明**：
  1. **官方事实核实（全部取自官方源码/实测）**：NPR = `model(x).sigmoid()`、高值偏 Fake、官方阈值 0.5；预处理开关 `no_resize`/`no_crop`（整图配方）与 ImageNet 归一化；仓库根**无 LICENSE**。UnivFD = CLIP ViT-L/14 + 线性头（`fc_weights.pth` 实测 4083 B = 768×1）、预处理 `CenterCrop(224)→ToTensor→CLIP 归一化`（**无 Resize**）、Real=0/Fake=1、仓库 MIT；主干 URL 内嵌 sha256 `b8cca3fd…`，**下载后自校验通过**（932,768,134 B）。两个仓库已克隆并**固定 commit**（NPR `781ced3f…`、UnivFD `030495ae…`）。
  2. **"不能照抄一个加载方式"用实测钉死**：`NPR.pth` 只能以 `state_dict['model']` 加载（裸 dict 缺 146 键），`model_epoch_last_3090.pth` 只能裸加载（无 `model` 键）。适配器按文件判定：先试用登记的方式，回退到另一种，**两种都不行就报错**；若回退的方式成功则明确提示"登记表需要更正"。
  3. **分数语义登记表**（`experts/score_semantics.json` + `utils/score_semantics.py`）：每专家记 权重身份/代码 commit/预处理（含**测量范围**枚举）/原始分数（字段、范围、方向、是否本地校准概率）/阈值口径/许可/评测状态。**又抓到一个语义陷阱**：noise 的 `raw_metric`（SRM 不一致度，越大越偏 Fake、无上界）与 `strength`（归一化后高=偏 Real）**方向相反**，登记表原来写错了字段，已改正并加回归测试——两个字段被当同一个用，就等于把这个专家反过来读。
  4. **统一评测入口** `scripts/g8_evaluate_experts.py`：一个入口测 PROBE/noise/NPR(A,B)/UnivFD，逐图记录原始分、耗时、状态与错误（错误如实入表，不丢不补零）；指标按登记表的方向统一（`higher_is_real` 用 1−score 算 AUROC，否则好专家会被读成坏专家）；**Mock 标记为不计入性能结果**；**test 分区拒绝阈值搜索**（阈值只能在 val 上选）。
  5. **官方奇数尺寸规则**：NPR 的 `x - interpolate(x,0.5)` 在奇数宽高上差一个像素（实测 327 vs 326 → 12/32 张报错）。按官方 README 的 AIGCDetectBenchmark 说明裁掉最后一行/列，并**逐次记录 `odd_size_cropped`**——改了输入一个像素，就必须说出来。
* **涉及/修改的文件清单**：`experts/score_semantics.json (Created)`、`utils/score_semantics.py (Created)`、`experts/npr.py (Created)`、`experts/univfd.py (Created)`、`scripts/g8_evaluate_experts.py (Created)`、`tests/test_score_semantics.py (Created — 17 项)`、`tests/test_g8_evaluate_experts.py (Created — 14 项)`、`plan.md (Modified — §3c)`、`.gitignore (Modified — 评测变体图像不入库)`、`sft_data/g8_eval/** (Created — 报告与逐图分数)`；外部：`/root/autodl-tmp/third_party/{npr,univfd}`（仓库 + 权重 + CLIP 主干）
* **执行结果与验证状态**：`core` 全绿、`module` **970 通过**。首批实测（CPU，val 8 源 × 4 格式）：**NPR-A** 整体 AUROC 0.516、acc@0.5 0.8125，**PNG 格 AUROC 1.00**（mean Fake 0.99996 / Real 6.6e-11），JPEG q70/q85 格 0.25/0.25（几乎全判 Real）；**NPR-B** 同形（PNG 1.00、JPEG 0.125/0.375/0.458）。noise 小批：方向 higher_is_real，AUROC 0.75（n=8）。**注意：这些都是 n=8 的管线验证，不是结论**；A/B 的取舍要用 val 全量（284 源）。
* **置信度或遗留待办（TODO）**：①**NPR A/B 全量 val 对比正在后台跑**（各约 10–15 分钟 CPU），结果出来按 val 选定 checkpoint 并写入登记表 ②UnivFD 与 PROBE 尚未评测（主干已就绪，CPU 也能跑 CLIP）③NPR 许可未明确——需要你去确认后才能谈研究之外的用法 ④本轮尚未做"生成 SFT 前的小批 Agent 人工验收"（属第 5 步）

### 2026-10-15 - 阶段 3：NPR 的 A/B 在 val 全量上选定（选定必须带证据）

* **当前操作动作**：按用户口径"val 负责专家选择"跑 NPR 两个 checkpoint 的 **val 全量**对比（各 284 源 × 4 格式 = 1136 张，错误 0），选定后把**证据一并写进登记表**；同时启动 PROBE 与 UnivFD 的 val 全量评测（GPU）。
* **对应计划锚点**：`plan.md` §3c
* **核心变更说明**：
  1. **选定 npr-a**：整体 AUROC **0.6363 vs 0.5870**；三个 JPEG 格一致领先（q70 0.470/0.393｜q85 0.512/0.457｜q95 0.564/0.507）；npr-b 仅在 PNG 格高 0.0012、acc@0.5 高 0.004（噪声量级）。**登记表里的 `selected` 不是光写一个名字**：带选定日期、分区、两边的完整数字与**诚实的保留**（见下）。
  2. **诚实的保留写进登记**：两者都弱，且**只在 PNG 格有效**（0.997–0.999 vs JPEG 格 0.39–0.56）——在本数据（Real=JPEG / Fake=PNG）上 NPR 是**格式受限**的检测器。它的价值在于 PNG 格的强信号与是否提供互补信息，且证据语义必须**带上适用格式条件**（阶段四的输入）。
  3. **测试随之改成"选择必须带证据"**：断言选定项含分区、两边 AUROC、四个格式格与保留说明——光写名字会让测试红。这正是"未选之前不能假装已选"那条测试的下一形态。
  4. **UnivFD 的设备/精度对齐**：官方 `.pt` 权重是 fp16，CPU 上 fp16 卷积又慢又未必支持 → CPU 跑转 fp32、GPU 跑保持 fp16 并整体搬到设备；不一致会导致 head 与特征的设备/精度错位。
* **涉及/修改的文件清单**：`experts/score_semantics.json (Modified — selected 带证据)`、`tests/test_score_semantics.py (Modified — 2 项改为证据断言)`、`experts/univfd.py (Modified — 设备/精度)`、`sft_data/g8_eval/*_npr-?_val/ (Created — 全量报告与逐图分数)`、`claude_operation_log.md`
* **执行结果与验证状态**：NPR A/B 全量 val 完成（1136 张 ×2，0 错误）；`tests/test_score_semantics.py` **18 通过**。**PROBE 与 UnivFD 的 val 全量评测在 GPU 上运行中**，结果出来补齐四专家对照表。
* **置信度或遗留待办（TODO）**：①A 的优势（0.05 量级 AUROC）只在单一 val 分区上验证过；若后续全量评测推翻，登记表按同样格式记录"改选"与理由 ②四专家表还缺 PROBE/UnivFD ③NPR 许可未明确，仍需用户确认 ④尚需判断"互补信息"——要等四专家数字齐了才能谈（错误重合度、成本）

### 2026-10-15 - 阶段 3：四专家在 val 全量上评测完毕，增益用配对 bootstrap 判定

* **当前操作动作**：跑完四专家（PROBE / noise / NPR-A / NPR-B / UnivFD）在 **val 全量**上的评测（各 284 源 × 4 格式 = 1136 张，错误 0），并把"新增专家是否提供互补信息"做成可复现的统计判定（错误重合 + 等权平均的配对 bootstrap），而不是看单专家指标拍脑袋。
* **对应计划锚点**：`plan.md` §3c（第 3 步"确定实际保留集合"）
* **核心变更说明**：
  1. **单专家（val，1136 张）**：PROBE 整体 AUROC **0.981**、acc@0.5 0.929，**四个格式格 0.975–0.988（不挑格式）**，GPU 74.8 ms/图；UnivFD **0.849**（0.817–0.893），38.7 ms/图且是最便宜的一个；noise **0.660**（0.624–0.701），63.3 ms/图；NPR-A 0.636（**PNG 0.997**，JPEG 格 0.470–0.564），271 ms/图；NPR-B 0.587。
  2. **错误重合**（同时错的频率 vs 独立预期）：npr-a×probe 0.023 vs 0.027、probe×noise 0.016 vs 0.027 → 错误互补；probe×univfd 0.026 vs 0.025 → 未显示；npr-a×univfd 0.275 vs 0.133 → 冗余。
  3. **增益判定（等权平均的配对 bootstrap，2000 次重采样，按图像配对）**：**npr-a**：整体 **+0.0061** CI[+0.0029,+0.0102] P(Δ≤0)=0.000，**PNG 格 +0.0223** CI[+0.0096,+0.0387] P=0.000，JPEG 格持平（CI 跨 0）→ **成立，且增益集中在 PROBE 最弱的 PNG 格**；**univfd**：整体 +0.0032 CI[−0.0026,+0.0090] P=0.135、PNG +0.0091 P=0.071 → **未成立（倾向正、从不伤害）**；**noise**：整体 **−0.0070** P(Δ≤0)=0.976、各格皆负 → **拖后腿**。
  4. **等权、不拟合**：组合规则是固定的等权平均，没有任何在 val 上拟合的参数——因此它可以被原样冻结，不需要再担心"在测试集上选参数"。
  5. 新增 `scripts/g8_gain_bootstrap.py`（配对 bootstrap，可复现），产物 `sft_data/g8_eval/gain_bootstrap_val.json` 与 `four_experts_val_comparison.json`。
* **涉及/修改的文件清单**：`scripts/g8_compare_experts.py (Created)`、`scripts/g8_gain_bootstrap.py (Created)`、`tests/test_g8_compare_experts.py (Created — 7 项)`、`sft_data/g8_eval/{four_experts_val_comparison.json,gain_bootstrap_val.json} 与 5 份全量报告 (Created)`
* **执行结果与验证状态**：五份全量报告（probe/univfd/noise/npr-a/npr-b）各 1136 张、错误 0；对照与 bootstrap 均可复跑；`core`/`module` 全绿（含新增 7 项对照测试）。**未触碰 test 分区**。
* **置信度或遗留待办（TODO）**：①以上全部是 **val** 结果，冻结后仍需在 **test** 上做最终验收（用户口径）②建议的保留集合需用户确认（PROBE 保留主检测器；NPR-A 作 PNG 格条件性线索；UnivFD 便宜且从不伤害但增益未成立；**noise 建议退出**——提案 §4.1 明确允许"缺乏收益时可退出"）③这是**检测器层面**的数字；模型看到的是证据 token，不是平均分——证据接口与呈现属阶段 4

### 2026-10-15 - 阶段三收尾：按用户四条指正重做统计、冻结候选方案

* **当前操作动作**：用户对上一轮的统计与结论提出四条方法学指正（重采样单位、表述强度、选择偏差、结论外推范围）并给出保留集合决定与两条边界。逐条执行：重做源级 bootstrap、改写结论表述、冻结候选方案、留下字段审计。
* **对应计划锚点**：`plan.md` §3c（结论表 + 四条纪律 + 两条边界 + 阶段四输入）
* **核心变更说明**：
  1. **重采样单位改为源**：1136 张变体来自 284 个源，按图独立重采样把同一源的四份当独立样本、CI 约缩 √4。改为**按源聚类**（抽到源就连同它在本格内的全部变体）。单格式格内每源恰一张，本无此问题。重做后：`probe+npr-a` 整体 +0.0061 CI[+0.0025,+0.0108] P=0.000（**仍成立**）、PNG 格 +0.0228 CI[+0.0097,+0.0401]；`probe+univfd` +0.0030 CI[**−0.0092**,+0.0140] P=0.285；`probe+noise` −0.0070 CI[**−0.0215**,+0.0059] P(Δ≤0)=0.838。
  2. **两处表述被指正并改写**：①"UnivFD 从不伤害"**不成立**（CI 含负值）→ 改为"**未证明明确增益，也未排除损害**"；②"noise 拖后腿"同样**不显著** → 退出理由落在弱判别力（0.660）、格式敏感、成本与语义负担，并写明**负增益只对"与 PROBE 等权平均"这一种组合成立**。
  3. **成本结论被同设备计时推翻**：原先拿 NPR 的 CPU 271 ms 比 PROBE 的 GPU 74.8 ms 不成立；补跑 NPR-A 的 GPU 计时后，**它是最便宜的（30.9 ms/图）**，univfd 38.7、probe 74.8（计时范围都含预处理）。
  4. **候选方案冻结**：`sft_data/g8_eval/FROZEN_CANDIDATE_SCHEME.json`（角色/条件/成本/统计口径），并写死 **test 的打开前置：证据接口、调用规则、提示词三者定稿之后**；同时记录"固定等权 ≠ 没有选择偏差，val 上选型没问题，前提是冻结后独立验收，test 不得成为下一轮选型集"。
  5. **字段审计留档**：校准表按 `raw_metric` 建分箱（`utils/reliability.py::lookup` → `calibrated_likelihood`），而 token 同时带方向相反的 `strength`；`utils/evidence_ledger.py` 的数值核对**把两者任一都当作"模型引用的那个数"接受**——引用反向字段也能通过。写进登记表的 `field_consumers` 并作为阶段四的具体输入。
  6. **测试固化决定**：新增 7 项断言冻结方案里的角色、源级重采样口径、test 打开前置、PNG 容器注意事项、**禁止"从不伤害"式表述**、组合层面结论的适用范围、同设备计时。
* **涉及/修改的文件清单**：`scripts/g8_gain_bootstrap.py (Modified — 源级聚类重采样 + 报告写明单位与每源张数)`、`experts/score_semantics.json (Modified — 角色/使用条件/退出理由/成本/字段审计)`、`plan.md (Modified — §3c 结论与纪律)`、`sft_data/g8_eval/{FROZEN_CANDIDATE_SCHEME.json,gain_bootstrap_val_sourcelevel.json,20261015_timing_npr-a_gpu_val/} (Created)`、`tests/test_g8_evaluate_experts.py (Extended — 7 项)`、`claude_operation_log.md`
* **执行结果与验证状态**：源级 bootstrap 复跑通过；三份测试文件 **46 通过**；`module` 全绿。**test 分区仍未触碰。**
* **置信度或遗留待办（TODO）**：①下一轮按用户指令进入**阶段四接线**（证据接口/调用规则/提示词）与**小批 Agent 人工验收**，三者定稿后才打开 test ②NPR 许可未明确（仅本地研究）③noise 的历史实现与结果保留、不再进默认工具箱；其 GPU 成本未计时（已标注）

