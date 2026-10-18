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

### 2026-10-15 - noise 的字段说明更正 + 阶段四开工（4.1 正式入口切 Qwen3）

* **当前操作动作**：先按用户指正更正 noise 的分数语义（我此前只做消费者留档，没查斜率），再按用户五项开始阶段四，完成第 1 项。
* **对应计划锚点**：`plan.md` §3d（阶段四执行计划）、§3c（登记表）
* **核心变更说明**：
  1. **noise 的更正（用户对）**：代码是 `strength = 1/(1+exp(−s·(x−m)))`，对 `inconsistency`（= `raw_metric`）**单调递增** ⇒ 两个字段**同号**，专家内部没有翻转。真正相反的是**算法原始假设**（"不一致度 ⇒ 篡改 ⇒ Fake"）与**经验数据关联**（G2 与今日 val：高残差落在**相机/Real 端**，mean(Real) 0.61 vs mean(Fake) 0.39）。登记表改为：两个字段同向、**差别只在量纲**（[0,1] vs 无上界）；新增 `direction_history` 把"算法假设 / 经验关联 / 代码状态 / 本次更正"分开写清；测试从"断言方向相反"改为"断言同号 + 量纲不同 + 假设与经验分开"。该专家已退出候选主线，**只补历史说明，不再优化**。
  2. **阶段四计划 §3d**：五项按用户给定顺序写成可验收条目（入口切换／真正关闭参考后验与预计收益／精简证据包与原始分数检查／旧专家退出与提示词重写／小批 Agent 人工验收），并写死 **test 仍关闭**，直到证据接口、调用规则、提示词三者定稿。
  3. **4.1 正式入口切 Qwen3**：`--mllm qwen` → `Qwen3VLClient`（固定 revision）；旧客户端移到 `--mllm qwen25`，**标注仅历史复算**（与 `FORENSIC_HALTING_POLICY=v2` 同性质）。顺带修三处：①PROBE 的 mock/real 判据原写作 `mllm_type != "qwen"`，会让历史客户端拿到 **mock 探测器**；②`main.py` 顶层导入 torch 与旧客户端，**CPU 路径一直为 GPU 栈付费**（正是惰性导出要避免的回归，在入口处又出现了一次）——现在导入 main 既不加载 torch 也不加载 transformers；③启动打印假定客户端必有 `revision`/`identity()`，改为有则打印、无则降级。
* **涉及/修改的文件清单**：`experts/score_semantics.json (Modified — noise 字段与 direction_history)`、`tests/test_score_semantics.py (Modified — 2 项改为同号/量纲/假设-经验分离断言)`、`plan.md (Modified — §3d 阶段四计划)`、`main.py (Modified — 客户端切换 + 惰性导入 + 身份降级)`、`tests/test_g8_entry.py (Created — 7 项)`、`claude_operation_log.md`
* **执行结果与验证状态**：`core` **413 通过**（新增 7 项入口测试）；`python main.py --image <Midjourney 图>` CPU mock 跑通；`import main` 后 `torch`/`transformers` 均未加载。**未运行 GPU**（入口切换的 GPU 验证放在 4.5 小批验收里）。
* **置信度或遗留待办（TODO）**：①4.2 真正关闭参考后验与预计收益（控制器仍加载校准表）②4.3 精简证据包与原始分数检查（含 field_consumers 审计里的那个缺陷）③4.4 旧专家退出注册 + 提示词按登记表重建④4.5 小批 Agent 人工验收⑤`--mllm qwen25` 的 GPU 复算未在真机跑过（属历史路径，必要时再验）

### 2026-10-15 - 阶段四 4.2：真正关闭参考后验与预计收益（并把它欠的检查补上）

* **当前操作动作**：按用户第 2 项执行"真正停止计算与提示注入，而不是只在提示词里藏字段"。做完发现关闭动作**会静默削弱证据检查**，于是同一项里把方向权威从校准表换到分数语义登记表（4.2b）——这正是提案 §7 警告过的那种"因校准字段消失而静默跳过"。
* **对应计划锚点**：`plan.md` §3d 的 4.2
* **核心变更说明**：
  1. **开关按运行策略求值，不是 import 时常量**：第一版写成 `config.USES_REFERENCE_POSTERIOR` 常量，结果 v2 复算的测试（fixture 在运行中切策略）也丢了校准表——48 个测试红。改为 `config.uses_reference_posterior(policy)` 函数，控制器在 `__init__` 里按当时的 `HALTING_POLICY` 求值，v3 策略改由参数 `compute_reference` 接收（跨模块不共享常量）。
  2. **v3 一次都不算**：校准表不加载（`ReliabilityTable.load()` 只在参考路径调用）、token 不再带 `calibrated_likelihood`/`reliability`/`applicability`/`semantics_aligned`、结案不再有 `posterior`/`tool_utilities`/`conflict_score`/`redundancy_discounts`、`_expert_weights()` 直接返回空（没有读者就不读表）、v3_signals 不再写参考派生键。**字段是"不出现"，不是写 0.5/0.0**——缺省值看起来像算过。
  3. **方向权威换到登记表（4.2b）**：整流器的优先级变为「登记表 → 校准表 → 专家自述」，并新增 `direction_rules`（逐专家：分数 → 方向，`Uncertain` 是合法结果）。**用户那条边界就此可执行**：NPR 低分判 `Uncertain`（"没看到它擅长的东西"≠"Real 证据"），PROBE/UnivFD 低分才判 Real。`prefer_registry` 是**调用方的策略决定**，不是默认值——v2 复算必须走校准表才能复现它产出的批次。
  4. **测试按新契约重写**：v3 的三条会话语义测试改为"没有参考块/没有参考建议"，且保留"把后验 patch 成极端反对方"——用来证明**这个值根本不会被咨询**；一组参考机制的测试显式挂 `v2_policy`（它们测的是 v2 机器，不是默认策略）；新增 8 条定点测试（低分 NPR→Uncertain、低分 PROBE→Real、整流器优先级、v2 不回退登记表、未登记专家不动）。
* **涉及/修改的文件清单**：`config.py (Modified — uses_reference_posterior 函数)`、`state_machine/controller.py (Modified — 运行时求值 + 四处写入分流 + _build_counters 显式传参)`、`state_machine/halting_v3.py (Modified — compute_reference 参数 + docstring)`、`state_machine/evidence_rectifier.py (Modified — 三级权威 + 共用的纠正方法 + 登记表取值)`、`utils/evidence_consistency.py (Modified — 认 score_semantics 方向来源)`、`experts/score_semantics.json (Modified — direction_rules + 说明)`、`utils/score_semantics.py (Modified — direction_for_value + 校验)`、`tests/{test_halting_v3,test_controller,test_controller_v3,test_probe_wiring,test_score_semantics}.py (Modified)`、`claude_operation_log.md`
* **执行结果与验证状态**：`core`/`module`/`full` 全绿（**full 1235 通过 / 3 跳过**）；CPU mock 会话复核：`posterior`/`tool_utilities`/`calibrated_likelihood`/`reliability`/`applicability` 在 v3 会话里**出现次数为 0**。**未运行 GPU**。
* **置信度或遗留待办（TODO）**：①4.3 精简证据包（模型可见 vs 后台完整）与原始分数检查（含账本"任一字段都接受"的缺陷）②4.4 旧专家退出注册 + 提示词按登记表重建③4.5 小批 Agent 人工验收④v2 复算的"逐字节一致"尚未用真实批次验证（本轮只保证计算路径仍在；冻结批次复跑留待需要时做）

### 2026-10-15 - 阶段四 4.3：精简证据包与"模型看到过哪些字段"的引用检查

* **当前操作动作**：按用户指令做 4.3——模型可见的精简包与后台完整记录分开，并修掉账本"任一字段都接受"的缺陷。
* **对应计划锚点**：`plan.md` §3d 的 4.3
* **核心变更说明**：
  1. **`EvidenceTokenizer.model_visible()`**：模型可见字段固定为 9 个（证据 ID／专家／测量名／执行状态／测量范围／**一个主分数 + 范围 + 方向 + 阈值 + 是否本地校准概率**／读数／产物与其含义）。`strength`、`raw_metric`、`support`、校准似然、适用性标签等**一律只在后台**——一个 token 曾经同时带五个可能互相打架的数值（G2-e 实测 frequency_v2 在 64 个样本里 15 个自相矛盾）。
  2. **适用条件随分数走**：读数里**先放判定这个值所依据的那条规则说明**，再放该专家的登记条件。于是 NPR 的包自带"在本次 val 的 PNG 编码条件下强、JPEG 低分不等于强 Real"，UnivFD 自带"中心裁剪、看不到边缘"——限制不再躺在一段没人更新的提示词里。
  3. **投递形态盖在 token 上**（`model_delivered: slim|full|none`）：账本、转换与审计据此判断"模型当时能引用什么"，无需调用方传参；**归档 trace 因此能自证模型看到了什么**。没有戳的 token 按旧批次处理（当时投递的就是完整 token），不倒扣。
  4. **账本新增 `undelivered_field_reference`（defect）**：引用了**没投递给它**的字段值——数字可能千真万确，引用仍不成立。这正是 noise 的 `strength`/`raw_metric` 场景（量纲不同、方向不同）。命中投递字段是正常路径，不受影响。
  5. **mock 客户端同步认两种形状**：包一改，三条冒烟检查立刻红——因为 mock 还在按旧字段解析、于是"后续轮引用数值"落空。它必须跟着活协议走，否则管线自检会在协议变更时**假装通过**。
* **涉及/修改的文件清单**：`state_machine/evidence_tokenizer.py (Modified — model_visible + 字段集合常量)`、`state_machine/controller.py (Modified — 两处注入改用精简包 + 投递形态盖戳)`、`utils/evidence_ledger.py (Modified — delivered 字段 + _delivered_fields + 新缺陷类型)`、`mllm/mock_client.py (Modified — 两种形状归一化)`、`tests/{test_evidence_pack.py(new,27),test_evidence_ledger.py,test_controller.py,test_g8_smoke.py} (Modified/Created)`、`claude_operation_log.md`
* **执行结果与验证状态**：`core` **452 通过**、`module` **1032 通过**、`full` **1274 通过 / 3 跳过**；实测会话：模型看到的字段为那 9 个，完整 token（20 字段，含 strength/raw_metric/support）仍在证据链里。**未运行 GPU。**
* **置信度或遗留待办（TODO）**：①4.4 旧专家退出注册 + 提示词按登记表重建（提示词现在还在描述旧工具箱与旧 token 形状，属于"接线未完成"）②4.5 小批 Agent 人工验收（全接通后才跑 GPU）③转换器（`utils/final_v2.py`）对新 token 形状（`model_delivered`、精简包）的适配在 4.4 一并检查

### 2026-10-15 - 阶段四 4.4：工具箱换代（注册表 + 提示词按登记表重建 + 缺席/失败如实呈现）

* **当前操作动作**：按用户指令完成 4.4——旧专家退出注册、提示词按登记表重建；并把"工具缺席/执行失败"从静默与崩溃改成如实呈现。
* **对应计划锚点**：`plan.md` §3d 的 4.4
* **核心变更说明**：
  1. **默认工具箱＝val 选定的三个**：probe / npr / univfd（`CURRENT_TOOL_TAGS`），jpeg/frequency_v2/noise 退出默认注册但**保留派发与解析**（v2 复算与旧 trace 需要）。CPU 侧补了 `MockNPRExpert`/`MockUnivFDExpert`，并把两个阶段三的评测器桥接成专家接口（`NPRExpert`/`UnivFDExpert`）：一条测量路径、两种调用方。
  2. **提示词由登记表生成**：工具动作行、量程、阈值、方向、测量范围、适用条件全部来自 `experts/score_semantics.json`，手写表只留给退役工具。**顺带抓到两件事**：①提示词"如何读 token"那段仍在描述**模型已经收不到的字段**（`calibrated_likelihood`/`applicability`）——4.3 改了投递、提示词没跟上，属真不一致；②我最初把 NPR 的内部说明（bootstrap CI、"适用条件写…不写…"）直接喂给了模型 ✗ → 登记表拆出 `prompt_note`（给模型的一句话）与 `usage_conditions`（内部，含统计与写法备注）。
  3. **解析器认新标签**：`Parser.extract_all_calls` 原先只认 probe/freq/noise/jpeg，**新的 npr/univfd 根本解析不出来**（产品缺陷：整条新链会"0 次调用"跑到底）；`CALLABLE_TOOLS` 改为由模式表派生，并加测试要求它覆盖登记表里的每个专家。
  4. **缺席与失败如实呈现（4.4b）**：调用了未注册工具 → 系统消息告知"本次会话不提供 X，可用工具：…"并计入 `unavailable_tool_calls`（旧行为是裸 `continue`，"没有这个工具"和"模型没调用"在 trace 里长得一样）；专家执行抛错 → 不再崩会话，留一条 `status="failed"` 的 token（含错误类型与原因）＋告知模型"这不是中性测量"＋计入 `failed_tool_calls`。失败 token 在账本里"投递零字段"，引用它给出的任何数字即 `fabricated_value` ✓（与 4.3 的字段规则接上）。
  5. **提示词哈希重钉**：`84bc2c766c6d712a` → **`b6d8a47c514dfe6c`**（进 VERSIONS/指纹，属必须记录的变更），并在测试里写明两次变更的原因。
  6. **"换工具箱会牵动所有编码了旧工具箱的地方"**：16 处测试随之更新（注入模式、mock 工具选择、控制器 fixture、提示词断言、解析器覆盖、增益对照的 dry-run 工具箱、端到端管线 fixture），其中折减审计那条改为**显式脚本驱动退役工具**——它验证的是 v2 可复算性，不该依赖当前 mock 的脚本。
* **涉及/修改的文件清单**：`mllm/message_builder.py (Modified — 登记表驱动 + 新读 token 契约 + 哈希变化)`、`mllm/mock_client.py (Modified — 工具选择换当前工具箱 + 两种 token 形状)`、`utils/parser.py (Modified — npr/univfd 标签 + CALLABLE_TOOLS 派生)`、`state_machine/controller.py (Modified — 注册表 + 缺席/失败处理 + 两个计数器)`、`main.py (Modified — 新工具箱接线 + 惰性导入)`、`experts/{npr,univfd}.py (Modified — 专家桥接 + Mock)`、`scripts/{g8_smoke,qwen_gain_baseline}.py (Modified)`、`experts/score_semantics.json (Modified — prompt_note)`、`tests/{test_toolbox(new,10),test_tool_absence(new,8),test_message_builder,test_expert_admission,test_controller,test_pipeline,test_probe_wiring,test_mock_mllm,test_qwen_gain_baseline,test_g8_entry,test_g8_smoke} (Modified/Created)`、`claude_operation_log.md`
* **执行结果与验证状态**：`core` **470 通过**、`full` **1292 通过 / 3 跳过**；CPU mock 会话用新工具箱跑通（probe 被调用、model_concluded）；提示词里出现的每个专家都在登记表中（测试锁定）。**未运行 GPU**——按用户口径，小批 GPU 验收放在 4.5。
* **置信度或遗留待办（TODO）**：①4.5 小批 Agent 人工验收（真实 Qwen3-VL + 新工具箱 + 最小审阅视图）②`utils/final_v2.py` 对新 token 形状（`model_delivered`、精简包）的适配仍需检查（转换器读 token 的地方）③NPR/UnivFD 的真机路径（GPU）尚未在会话里跑过——它们只在阶段三的离线评测里跑过④退役工具的代码与结果保留，未删除


### 2026-10-15 - 阶段四 4.5a：小批前的 CPU 核查（用户点名的三项，逐项先复现再修）

* **当前操作动作**：按用户小批前的两项要求与一项提醒做 CPU 核查与修复——①转换器适配精简包（账本识别新字段、原始 `report` 不丢失、训练对话不重新注入后台字段）②禁止真实入口静默降级为 Mock ③打印实际模型输入以确认没有把 bootstrap 统计/内部写法要求塞回包里。**未跑 GPU，未加载任何真实权重**（真实入口的 CUDA 门禁在加载权重**之前**）。
* **对应计划锚点**：`plan.md` §4.5a（本次新增）、§5（证据包边界）、§3d 的 4.5；4.4 条目的 TODO ② `utils/final_v2.py` 对新 token 形状的适配
* **核心变更说明**：
  1. **账本读不懂精简包（最严重的一处，先复现）**：运行时喂账本的是证据链上的完整 token，而转换器 `audit_dialogue` 喂的是 trace 里**当时投递的那段 JSON**——阶段 4.3 起即精简包。包用 `expert` 而账本读 `source`，于是整条记录被跳过：同一句完全正确的引用（"PROBE 的分数 0.93 指向 AI 生成"）被判成 `fabricated_expert_result`（DEFECT）。**新主线的每个样本都会倒在审核这一关**，而且是在样本落盘之后才被发现。修法：新增 `ledger_view()` 把投递记录翻成内部形状（`expert`→`source`、`score.value`→**登记表指定字段**而非写死的 `raw_metric`、`reading`→文本字段、`scope`/`status` 直通、方向取自 `direction_for_this_value`＝登记表逐值规则），`_delivered_fields` 继续按投递戳算"模型能看到什么"。`direction_basis` 新增 `registry_rule` / `registry_rule:no_direction` 两类，冲突判定沿用 CANDIDATE（交人工），并在 JSON 里点明来源与旧 support 标签不同。
  2. **账本认不出新工具的名字**：`SOURCE_WORDS` 里只有 probe/noise/jpeg/freq。模型写"NPR 的分数是 0.93"时，**整句话对账本不可见**——不查数字、不查方向、不查编造。已按提示词给模型的写法补 `npr`/`univfd`（npr: npr/邻域/上采样/upsampling/neighboring pixel；univfd: univfd/univ-fd/clip/线性头）。另加一条：数值不承载方向（`Uncertain`）而句子却断言了方向时 → `direction_from_neutral_value`（CANDIDATE），正是用户 2026-10-15 那条边界（JPEG 下的 NPR 低分不得读成强 Real）。
  3. **失败包带着"带方向的空分数槽"**：`model_visible` 对失败 token 仍给出 `score`（`value: None`，却带 `range`/`direction`/`threshold`）——模型会把它读成一次中性测量。失败包现在**没有分数、没有尺度与方向**，只有状态与"这不是中性结果，也不是证据"。
  4. **状态小结把后台字段放回模型输入**（打印实际输入时当场发现，见第 8 条）：`_state_note` 无条件写 `P(Fake)=…／raw_metric=…／适用 …`；完整投递下没错，精简投递下正好是被包收起来的那些字段，而且**与包冲突**（包说 AI-generated，同一轮小结说"方向 —"）。改为按投递戳说话（`_state_line`）：slim 写"分数=…（该值的方向：…）"，旧会话照旧。
  5. **登记表混着两类读者**：`prompt_note` 与内部说明（`usage_conditions`/`limits`/阶段编号）同格。noise 的"退役…仅供回放"、`chosen_on` 里的 `G2 校准集`、以及段头里的 `the G7 fitting group of 62 sources` 都会随提示词送到模型面前（工具段现在是登记表生成的，即 4.4 引入的路径）。已按字段分家（模型侧：`display_name`/`prompt_note`/逐值 `prompt_note`；内部侧：`role`/`retirement`/`usage_conditions`/`limits`/`note`），空的手写段不再留下段头，并加**机械扫描**防回归。
  6. **失败后继续的会话把训练对话错位**：4.4 起失败记录也进证据链但**没有投递轮**，而转换器按**次序**把投递轮对回 token——下一个 token 会被顶到上一轮的位置，训练数据里出现模型当时没拿到的测量。改为按 `evidence_id` 认领；认不到时（旧 trace／id 变过）退回"第一个**可投递**的 token"，跳过失败与从未投递的记录。
  7. **训练对话重新注入后台字段 + 模型 report 丢失**：转换器用 `to_json(token)` 注入完整 token（`strength`/`raw_metric`/`calibrated_likelihood`/`reasoning`），并按投递形状改为 `model_visible`（旧批次保持完整 token，回放一致）。另：推理提示词要求 `<verdict>` JSON 带 `report`，而渲染的答案里**没有这个字段**——训练出的模型被教成写一个不含 report 的判决，与自己的说明相反。答案现在带上**被采纳那一轮**的 `report` 原文（`adopted_report`，与 reasoning 同源；那一轮没写就不含此字段，绝不代写）。
  8. **真实入口不得静默降级为 Mock**：原写法是 `print("Warning: CUDA not available. Falling back to MockMLLMClient.")` 然后继续跑。改为 `require_cuda_for_real_sessions()`：在**加载任何权重之前**直接抛错，并说明 CPU 该怎么跑（显式 `--mllm mock`）。PROBE 权重缺失时不换 Mock（不会造假分数），但警告明确写出"本次会话不提供主检测器，结果不得按 g8 工具箱口径解读"。
  9. **证据（用户要的"打印几份实际模型输入"）**：新增 `scripts/g8_dump_model_inputs.py` → `sft_data/g8_model_inputs/<时间戳>/model_inputs.md`，含系统提示词原文、一次真实会话的全部对话（CPU/mock）、逐值证据包（NPR 低分＝`Uncertain` 的边界、失败包）与内部标记词表扫描；命中即退出码 1。当前扫描 **0 命中**。
* **涉及/修改的文件清单**：`utils/evidence_ledger.py (Modified — ledger_view/is_slim_pack/_registry_field、SOURCE_WORDS 补 npr/univfd、_token_direction 认逐值规则、direction_from_neutral_value)`、`state_machine/evidence_tokenizer.py (Modified — 失败包无 score；逐值规则只用 prompt_note 且补 direction_for_this_value)`、`state_machine/controller.py (Modified — `_state_line` 按投递形状写状态小结)`、`utils/final_v2.py (Modified — 按 evidence_id 认领投递轮、按投递形状重建 payload、adopted_report 带上模型原文)`、`main.py (Modified — require_cuda_for_real_sessions + PROBE 缺席措辞)`、`mllm/message_builder.py (Modified — 提示词字段清单补 `direction_for_this_value`/`threshold_chosen_on`/`artifact_meaning`，空的手写段不再留段头，提示词哈希重钉)`、`experts/score_semantics.json (Modified — 逐值规则加 prompt_note、noise 的模型可见文字与 chosen_on 去内部措辞、两类读者写进契约字段)`、`scripts/g8_dump_model_inputs.py (Created)`、`tests/test_slim_conversion.py (Created — 端到端：会话→转换→样本)`、`tests/test_score_semantics.py`、`tests/test_evidence_pack.py`、`tests/test_message_builder.py`、`tests/test_controller_v3.py`、`tests/test_g8_entry.py`、`tests/test_tool_absence.py`、`tests/test_final_v2.py`、`README.md (Modified — 入口不降级 + 3.7c 打印输入)`、`plan.md (Modified — §4.5a)`、`claude_operation_log.md`
* **执行结果与验证状态**：`core` **495 通过**；`full`（全量）**1318 通过 / 3 跳过**（本次改动前为 1292 通过 / 3 跳过，新增 26 条为本轮的新测试）。`prompt_digest` = `af7003e17e0d6a42`（提示词哈希 `b6d8a47c514dfe6c → 8143461c167d31ed → a01272f96bbd66c7`，测试里写明两次变更的原因）。`python scripts/g8_dump_model_inputs.py` 退出码 0、内部标记 0 命中（机器词表判定）。端到端新测试：CPU 上真跑一次含执行失败的会话 → 转换 → 样本，断言审核缺陷为 0、训练对话字段 ⊆ 模型可见字段、交付轮 id 是被投递的那个（不是失败记录）、模型 report 原文在位、旧投递（无戳/`full`）与从未投递（`none`）三种旧行为未变。**未跑 GPU。**
* **置信度或遗留待办（TODO）**：①字段名引用检查仍以**数值**为判据：模型写"PROBE 的 strength 是 0.93"（数值确实投递过、名字没有）目前不判缺陷——已知边界，未扩大改动③`utils/final_v2.py` 对旧批次的整流语义（`rectified` 只在值上，id 不变）未逐批核对，靠"认不到 id 时退回首序可投递 token"兜底④小批（8–16 源 × 4 格式）与人工查看原始图像/证据包/完整对话仍待用户授权后执行⑤批次名 `b1` 尚未得到用户确认

### 2026-10-15 - 阶段四 4.5c：真机冒烟抓到两个必崩的 bug（UnivFD 的 np 与三专家的设备），并补上小批入口

* **当前操作动作**：用户开通 GPU 后按 §4.5 的顺序推进——先跑真实冒烟（并把冒烟工具箱扩到能点齐 probe/npr/univfd），再逐个专家在真机上打一遍分，然后写小批入口。**两个产品级 bug 都是真机跑出来的，不是推演出来的**；每个都先复现、再修、再留机械检查。
* **对应计划锚点**：`plan.md` §4.5（小批）、§4.5b（生成入口）；4.4 条目的 TODO ①（4.5 小批验收）
* **核心变更说明**：
  1. **冒烟六项全绿**（真实 Qwen3-VL + probe/npr/univfd + 真实权重，固定 revision、贪心、非思考）：图像进多模态消息、一次合法调用被执行、结果被后续轮引用（引用的是 `aigi_discriminator_score`）、结案可解析且收尾轮仍有图像 token、三类记录分别留存、首轮视觉观察 103 字且另列疑似异常。**耗时 70.5 s｜峰值显存 18.48 GB**。产物 `sft_data/g8_smoke/20261015_211332/`。
  2. **冒烟原先点不齐工具箱**：`--experts` 只认 `probe` 与一个固定值桩（桩的注释还写着"注册在 noise 标签下，因为提示词只认旧四家"——4.4 早就改了）。现在 `npr`/`univfd` 都能点（mock 客户端则用各自的 mock 专家），并加测试要求"冒烟必须能点齐 `CURRENT_TOOL_TAGS`"。
  3. **真机 bug ①：UnivFD 一调用就崩**。`experts/univfd.py::_score` 用了 `np` 而模块里没导入（`NameError`）。在小批里它会被 4.4 的失败包装记成一次 `failed` 工具调用、会话照跑、数据照收——正是"看起来跑通"。修复并加测试：用一个假 evaluator 顶替 CLIP，让**这条路径真的被执行一遍**（无需 1.7 GB 主干）；这正是 `test_external_experts.py` 开头那句"没权重也能测的部分，恰好是加载权重时才暴露的部分"。
  4. **真机 bug ②：三个专家的默认设备是 cpu**。`ProbeDinoExpert()`/`NPRExpert()`/`UnivFDExpert()` 的 `device` 默认 `"cpu"`，而 `main.build_pipeline`、冒烟、生成入口都不传 device——整批数据会**在 CPU 上量出来**，与登记表里那次 `--device cuda` 的 val 数字不是同一个测量条件；更糟的是 UnivFD 的官方 `ViT-L-14.pt` 在 CPU 上根本跑不了（实测三种走法都失败：输入 fp32、输入 fp16、主干整体 `.float()`）。修法：`experts/device.py::resolve_device`——显式给了就用，没给就 cuda→cpu；三个 evaluator 与专家统一走它；**批次元数据与启动横幅都把解析后的设备写出来**（"同一权重、不同设备"是会悄悄改数字的那类差别）。UnivFD 在无 CUDA 时 `unavailable_reason()` 明说"官方主干无法在 CPU 上运行"，而不是跑到一半抛异常。
  5. **真实会话不再静默缩减工具箱**：`build_pipeline` 原先只检查 PROBE 是否可用，npr/univfd 缺权重会一路跑到调用时抛异常。现在真实会话里它们不可用即**拒绝并说明**（"真实会话不缩减工具箱"）——少一个专家不是"少一次调用"，而是这批数据的口径变了。
  6. **小批入口 `scripts/g8_generate_batch.py`**（§4.5b）：会话由 `main.build_pipeline` 构造（与验收同一条管线：CUDA 门禁、专家不可用即拒、设备自动解析），客户端只建一次、每条会话新建状态机（证据链/计数器不跨会话泄漏）；`assert_not_frozen_batch` 冻结门禁；`--dry-run` 走 mock 且**不写批次目录**（mock 会话的记录与真实会话长得一模一样，混进去就是"假数据当真的收"）；已存在的记录跳过（可续跑）；记录按 g7 的记录契约写全（`record_schema`/`evidence_check`/`verdict_status`/`trace_path`/五个计数器），少一个字段下游就静默少一段。
  7. **解码条件走同一条入口**：`build_pipeline(greedy=…, inference_mode=…, local_files_only=…)`——批次与冒烟都贪心，不让调用方自己造一个"等价客户端"（那正是"我看的"与"跑的"分叉的地方）。
* **涉及/修改的文件清单**：`experts/device.py (Created — resolve_device)`、`experts/external_base.py`、`experts/npr.py`、`experts/probe_dino.py`、`experts/univfd.py (Modified — 三处设备解析 + np 导入 + CPU 不可用说明)`、`main.py (Modified — build_pipeline 解码参数 + 真实会话拒绝缩减工具箱)`、`scripts/g8_smoke.py (Modified — 工具箱可点齐三个真实专家)`、`scripts/g8_generate_batch.py (Created)`、`tests/test_external_experts.py`、`tests/test_g8_smoke.py`、`tests/test_g8_batch_entry.py (Created)`、`plan.md (Modified — §4.5b)`、`claude_operation_log.md`
* **执行结果与验证状态**：GPU 冒烟六项通过（70.5 s｜18.48 GB）；逐专家真机打分：probe Fake(png) 0.9999/Real(jpg) 0.0170、npr Fake 1.0000/Real 0.0000（**Real 低分判 `Uncertain`**，与登记表的 JPEG 边界一致）、univfd Fake 0.7569（修复后）；`core` **510 通过**；批次入口 CPU 干跑 2 条通过（干跑隔离目录 `traces/dry_run_records/`，批次目录零污染）。**产物**：`sft_data/g8_smoke/20261015_211332/`（报告 + 会话 + `expert_gpu_check.json`）。
* **置信度或遗留待办（TODO）**：①UnivFD 的 CPU 不可用是我方实测结论（三种走法全失败），官方是否另有 CPU 走法未查（我们只需要 GPU 路径）②三个专家的 GPU 计时只各打了一组图（冷启动 28.8 s → 热 1.8 s 是 probe 的两次调用，不是稳定基准）③小批 b1（8 源 × 4 格式 = 32 会话）正在跑，产物与记录见下一条④npr 的两种 checkpoint（npr-a/npr-b）在专家桥里固定为 npr-a（登记表 `selected`），本批未做 A/B

### 2026-10-15 - 阶段四 4.5：b1 小批跑完（8 源 × 4 格式），并产出最小审阅视图

* **当前操作动作**：在用户"GPU 已开启 + 批次名 b1 + 8 源 × 4 格式"的确认下跑完小批，产出审阅视图，并把两条候选发现核清。
* **对应计划锚点**：`plan.md` §4.5（小批验收）、§4.5b（生成入口）、§4.5a（小批前核查）；用户 2026-10-15 选定的批次名与规模
* **核心变更说明**：
  1. **变体与批次**：`build_variants_g4.py --generation g8 --batch b1 --sources 8`（类平衡、固定 seed）→ 8 源 × 4 格式（png / jpeg_q95 / jpeg_q85 / jpeg_q70）= 32 个变体；`scripts/g8_generate_batch.py` 跑完 32 条真实会话（Qwen3-VL@`0c351dd0`、贪心、非思考、固定 revision）。
  2. **结果**：判定 32/32 与真值一致（四格式各 8/8）；全部 `model_concluded`；引用检查 32 pass、**0 缺陷**；投递形状全部 `slim`；0 次失败工具、0 次缺席调用；均 27.5 s、最长 43.2 s、合计 880 s，峰值显存 18.1 GB。
  3. **必须报的形状问题**：**32/32 只调用了 PROBE，npr/univfd 一次都没被调用**。提示词把 PROBE 写成主判据、其他专家"只是佐证且可选"，模型就照办了——这与 g7 批次"PROBE 单专家优于 Agent"是同一个形状。后果：这批数据没有在模型闭环里检验两个新专家（它们各自的真机路径我在 4.5c 单独验过：NPR Real/JPEG 低分判 `Uncertain`、UnivFD 0.7569）。扩不扩生成、要不要改协议让模型更愿意取第二意见，交用户定。
  4. **两条候选发现逐句核过，都是账本假阳性**（不是模型的问题）：①"低于阈值 0.5"——阈值本来就在精简包的 `score.threshold` 里，模型引用它看到过的东西，账本却把 0.5 记成"只出现在自由文本里"；②"未被外部判别器识别为 AI 生成"——否定句被 `_asserted_direction` 读成"断言 Fake"，与 Real 方向判成冲突。已修：交付集合加入**阈值**（`_delivered_fields` 与 `ledger_view` 两侧一致，失败测量仍交付空集）、否定形态补入词表（未被/未能/未识别/未判定/不构成/不属于/不表明/不支持），并各加回归测试；**b1 记录保留原样**（记录是那次运行的产物，不回溯改写），修后重审这两条会话为 0 缺陷 0 候选。
  5. **最小审阅视图 `scripts/g8_review_view.py`**：直读 **记录 + trace** 出静态 HTML（`sft_data/g8/review/b1/index.html` + 32 个会话页），每页含原始图像（相对路径引用，不拷图）、**当时实际投递的包原文**（格式化后逐字呈现）、完整原始对话（含状态小结与系统提示）、结案、停止原因、计数器、引用检查。**不走训练准入**：`admit_trajectories_g4` 对本批 32/32 判 reject，理由全是旧世界（`unknown tool 'npr_expert'` ×32、要求无工具基线）——验收看行为、训练看准入，混起来审阅者会什么都没得看。
* **涉及/修改的文件清单**：`scripts/g8_generate_batch.py`、`scripts/g8_review_view.py (Created)`、`utils/evidence_ledger.py (Modified — 阈值交付 + 否定形态)`、`tests/test_g8_review_view.py (Created)`、`tests/test_evidence_ledger.py`、`tests/test_slim_conversion.py`、`plan.md (Modified — b1 结果与形状问题)`、`claude_operation_log.md`；**产物**：`sft_data/g8/variants/b1/`（32 变体 + 清单）、`sft_data/g8/trajectories/b1/`（32 记录 + `metadata.json`）、`traces/sft_sessions/g8/b1/`（32 会话 trace）、`sft_data/g8/review/b1/`（33 页）、`sft_data/g8/admission_report_b1.json`、`sft_data/g8_smoke/20261015_211332/`（冒烟报告 + 逐专家真机检查）
* **执行结果与验证状态**：b1 32/32 正确、32 pass 引用检查、全部精简投递（记录可自证）；`core` **重新通过**（新增假阳性回归 3 条、审阅视图 6 条）；GPU 冒烟六项通过（18.48 GB）；逐专家真机打分组见 4.5c。**未跑 test 分区、未解冻任何批次、未删除任何数据**（只删了自己几分钟前生成的 dry-run 记录与测试污染目录）。
* **收尾（同一条内补记）**：提交后发现**干跑的 6 个 mock trace 落在批次 trace 目录里**（记录的隔离先做、trace 的隔离漏了），已删孤儿文件并修根因（干跑的记录与 trace 一律同进同出，显式 `out_dir` 时同目录），并加测试守住；修后批次材料自洽：32 条记录引用的 trace 全部存在，trace 目录里没有不被引用的文件。
* **置信度或遗留待办（TODO）**：①形状问题（模型不调 npr/univfd）待用户决定是否改协议，不是 bug ②训练准入对新工具箱的对齐（`unknown tool`／无基线）属阶段 5 ③审阅视图是最小版：没有浏览器内保存决定、没有与无工具基线的对照（那要等基线臂）④批次尚未冻结——冻结前应先把形状问题定下来，免得冻结一批"只会调 PROBE"的数据

### 2026-10-15 - 阶段四 4.6：验收审核工作台（八条要求；b1 事实零改动）

* **当前操作动作**：按用户八条要求把最小审阅视图升级为可用的审核工作台，并按要求**只改展示与审核功能**：b1 的 32 条记录、32 份 trace、判定与批次元数据一律未动、未重跑模型（`git status` 对 `sft_data/g8/{trajectories,variants}` 与 `traces/sft_sessions` 零差异）。
* **对应计划锚点**：`plan.md` §4.6（本次新增，八条逐条落点写在里面）、§4.5 及其结果块
* **核心变更说明**：
  1. **存储与身份（第 1、2、3 条）**：新增 `utils/review_store.py`——决定＝`(代次, 批次, 样本内容指纹)`；指纹对**页面展示的事实**取（判定/停止原因/计数器/引用检查/逐轮原文，`sample_fingerprint`），**由 Python 算好嵌进页面**，页面里的 JS 只做字符串比对、不重算（两套实现必然漂移）。规则：内容不变 → 决定保留；内容变 → 标**已过期**、不继承；导入时 schema/代次/批次任一不符 → **整份拒收**；待修/拒绝没写原因、审核人空、分类不认识 → 拒收该条。台账是**追加式** `sft_data/<gen>/review/<batch>/dispositions.jsonl`（改主意留痕），写回只能经 `scripts/g8_review_import.py`；页面明写"浏览器保存 ≠ 已写回仓库"。按钮叫**「通过（本次审核）」**，页面与导入两侧都声明"不改可训练标记、不走训练准入"，也不复用 `sft_data/review/dispositions.jsonl`（那是训练口味的台账）。
  2. **原文可核对（第 4 条）**：区块改名「已记录的会话原文」；逐轮标来源 `任务下发／模型输出／工具返回（投递包）／控制器提示`（`classify_turn` 按 from 与内容判定，控制器提示**不再伪装成人类输入**）；JSON 除美化外**并给原始文本折叠**与**整份 trace 下载链接**；助手文本原样展示、不总结不补写——测试逐轮断言页面文本与 trace 逐字相同。
  3. **投递与后台分开（第 5 条）**：主区的"投递 N"用的是 **trace 里保存的那段 JSON**，不再调用今天的 `model_visible()` 重新生成（那是拿今天的代码改写过去的输入）；后台完整 token 移入折叠区；长内容截断但必给"完整内容"折叠。新增测试：把 trace 里的包换成一个**今天的代码不会产出**的形状（带 `legacy` 字段），页面必须原样显示。
  4. **审计信息（第 6 条）**：原图与**实际输入变体**并排、实际尺寸并标来源（trace metadata）、模型 revision/精度/推理模式与三个专家的类与**设备**（批次 `metadata.json`）、停止原因与 policy_reasons、请求/执行/重复/抑制/缺席/失败/格式与动作违规计数、**引用检查的具体发现文本**（kind+why+句子），缺失项一律写"未记录"。
  5. **日志完整性（第 7 条）**：核实结果是 b1 缺三项——**初始 system prompt 原文、每次调用的多模态输入、客户端内部重试**（`QwenVLClient.generate` 内建最多三次尝试并注入纠正，中间那次输出与纠正文本都不留）。页面给每批一张"日志完整性"表逐项标已记录/未记录，并明写"**不用今天的代码重建冒充原始日志**"。**后续批次补记**：客户端累积 `call_log()`（逐次尝试：原文、格式校验结果、注入的纠正、**那次真正发出去的完整提示文本** + sha256 + 图像数 + 输入统计）与 `system_prompt_record()`（原文+sha256）；控制器在结束时写进 trace metadata（`model_calls`/`system_prompt`）。空结构＝未记录（旧批次与替身客户端天然落在这一侧）。
  6. **验证（第 8 条）**：新增 4 个测试文件、共 66 条（另删掉被取代的 6 条；全量 1345 → 1405）——`test_review_store.py`（指纹稳定性与变更敏感性、原因必填、跨代次/跨批次/未知样本/错误 schema/训练用途声明一律拒收、台账追加语义、进度与过期统计）、`test_g8_review_workstation.py`（逐轮来源标注、**原文逐字一致**、投递包不被重生成、后台折叠、发现文本、图片链接可达、未记录标注、页面声明验收非训练、构建只写 HTML、导入命令拒绝跨批且只写验收台账）、`test_model_call_log.py`（重试逐次留痕、提示原文与哈希、reset 清空、trace 落地、无接口时写空而不是编造）、`test_review_page_js.py` + `tests/js/*.js`（**用 Node 跑页面自己那段 JS**：草稿刷新不丢、指纹变了标过期、导出带指纹与 `acceptance_only`、跨批导入被拒并给出提示、缺原因/缺审核人不落草稿）。
* **涉及/修改的文件清单**：`utils/review_store.py (Created)`、`scripts/g8_review_workstation.py (Created — 取代 g8_review_view.py)`、`scripts/g8_review_import.py (Created)`、`mllm/qwen_client.py`、`mllm/qwen3_client.py`、`mllm/base.py`、`state_machine/controller.py`、`utils/logger.py (Modified — 模型侧调用日志)`、`tests/test_review_store.py`、`tests/test_g8_review_workstation.py`、`tests/test_model_call_log.py`、`tests/test_review_page_js.py`、`tests/js/{review_page_stubs.js,review_page_assertions.js} (Created)`、`scripts/g8_review_view.py`、`tests/test_g8_review_view.py (Deleted — 被工作台取代)`、`plan.md (Modified — §4.6)`、`README.md (Modified — §3.7c 工作台与写回命令)`、`claude_operation_log.md`；**重建产物**：`sft_data/g8/review/b1/`（索引 + 32 会话页）
* **执行结果与验证状态**：`core` **通过**；全量 `full` **1405 通过 / 3 跳过**（1345 → 1405：新增 66，删旧 6）；b1 批次材料**零改动**（记录/trace/变体 `git status` 无差异，未重跑模型）；页面在真实 b1 上重建成功（32 页 + 索引），抽查逐轮原文与 trace 逐字一致、两张图链接可达、日志完整性表如实标出三项未记录；导入命令在 b1 上 `--dry-run` + 临时台账跑通（未触碰真台账）。
* **置信度或遗留待办（TODO）**：①页面**未做**人工浏览器实测（无浏览器）：规则由 Node 跑页面自身 JS + Python 镜像双向覆盖，但"在 Firefox 里点一遍"仍应由用户或后续会话做一次②b1 的三项日志缺口**不可补**（补了就是伪造原始日志）——要补的是下一批；本工作台已把新字段接上，下一批跑完即可看到"已记录"③审核决定只写验收台账，与训练准入完全分离；将来要把验收结论接到训练门禁上，需要用户先定规则（本阶段刻意不做）④审查者视角的便利项（键盘快捷键、按审核人筛选导出子集）未做，按需再加

### 2026-10-15 - 阶段四 4.6b：审核工作台的布局与展示（顶部固定栏、链接状态、四类调用状态）

* **当前操作动作**：按用户四条要求调整工作台的**布局与展示**：审核控件移到顶部固定栏、链接的访问状态与焦点样式、调用状态按四类如实展示。**未改动任何样本内容、证据或审核结论**：b1 的 32 条记录、32 份 trace、变体与批次元数据零改动（`git status` 无差异），未重跑模型，未动台账。
* **对应计划锚点**：`plan.md` §4.6（八条要求的展示层细化）；用户 2026-10-15 的四条要求
* **核心变更说明**：
  1. **顶部固定审核栏（要求一）**：`<header class="topbar">` 现在是页面 **body 里的第一个块**，`position: sticky; top: 0; z-index: 20; background: canvas`（不透明、跟随深浅色）；栏内依次是**批次目录链接／内容指纹／上一条·下一条**、样本名、小标签栏（判定/真值/结案来源/停止原因/格式）、以及**唯一一套**审核控件（决定徽标、审核人、分类、通过/待修/拒绝、导出、导入、提示）。底栏删除，`localStorage` 草稿、校验（待修/拒绝必填原因、审核人必填）、翻页与导入导出逻辑**未改**（同一批函数、同一套 id）。窄屏 `flex-wrap` 换行，备注框 `min-height: 38px; max-height: 96px`，正文不被遮挡（sticky 在文档流内）。
  2. **链接访问状态（要求二）**：`a:link` 蓝 `#0b5fb0`／`a:visited` 紫 `#8a3fb8`，深色主题各一套（`#74b9ff`／`#cba0ff`），`a:hover` 下划线、`a:focus-visible` 2px outline（深色下换色）。页面明写：**链接颜色只表示你打开过（浏览记录），与审核状态无关**；审核状态继续由独立标签呈现（`未决定/通过/待修/拒绝/已过期/草稿`）。
  3. **调用状态（要求三）**：新增 `call_status()`，把四类**各自拆开**并各自带 `recorded` 标志——`请求了不可用工具`（工具 tag＋对应专家＋轮次＋提示原文）、`工具执行失败`（专家＋异常原文＋轮次）、`重复请求被跳过`（次数＋专家＋轮次）、**未调用专家**（`available_experts` 减去已测量/不可用/失败）。两者皆空且已记录 → **"调用异常：无"**；键不存在 → **"未记录"**，并写明"不能当作没有异常"。原先那行 `缺席 []｜失败 []` 已删除——空数组既可能是"确实没有"也可能是"这个版本没记"，而"模型没调用某专家"是第三件事。b1 每条会话现在如实显示：调用异常无，**未调用专家：npr_expert、univfd_expert**（这正是 4.5 记下的形状问题，现在每页可见）。
* **涉及/修改的文件清单**：`scripts/g8_review_workstation.py (Modified — CSS/顶栏/调用状态)`、`tests/test_g8_review_workstation.py (Modified — 新增 15 条：顶栏结构、控件唯一性、链接状态、四类调用状态互不混淆)`；**重建产物**：`sft_data/g8/review/b1/`（索引 + 32 会话页）；`claude_operation_log.md`
* **执行结果与验证状态**：**重建前后指纹逐条比对完全一致**（32/32，重建只动排版），且页面里的指纹与 `utils.review_store.sample_fingerprint()` 独立复算一致；台账进度保持 0/32、`repo_decisions` 未被清空；真实页面上核对：顶栏是 body 第一个块、粘顶/不透明/层叠/换行/备注高度齐备、控件各 id 只出现一次、链接四种状态样式齐备、`调用异常：无` 与未调用专家列出、`缺席 []` 字样消失。`full` **1418 通过 / 3 跳过**（1405 → 1418：新增 13 条，另 2 条为既有用例的措辞对齐）；**未跑 GPU、未动批次材料与台账**。
* **置信度或遗留待办（TODO）**：①**滚动可见性与"正文不遮挡"是按 CSS 与结构核对的，没有真浏览器实测**（本机无浏览器）：sticky 在文档流内、顶栏为不透明背景且有 z-index，属标准用法；建议用户在浏览器里滚一次确认②索引页与会话页都有审核人输入与导出/导入（**同一份草稿与同一个审核人**，跨页一致），这是刻意的：索引页是批量视图、会话页是逐条视图③四类调用状态的"轮次"来自 trace 里对应的提示轮；若将来某类事件没有对应提示轮，会如实写"未记录"，不推测

### 2026-10-15 - 阶段四 4.6c：旧模型（Qwen2.5-VL-7B）下架核查与改名隔离

* **当前操作动作**：按用户要求核查 `psychology_video_project` 里的 Qwen2.5-VL 对当前项目是否还有影响、审核该目录内容、**先改名不删除**，并跑全套验证。**未删除任何文件**；改名是同盘 `rename`，改回即恢复。
* **对应计划锚点**：`plan.md` §3b（模型固定与历史复算入口）、`GENERATIONS.json`（g7 用 Qwen2.5-VL、g8 用 Qwen3-VL）；用户 2026-10-15 的指示
* **核心变更说明**：
  1. **依赖面（逐处列出，不是印象）**：当前项目里只有三处读那条路径——`config.QWEN_MODEL_PATH`（常量）、`mllm/qwen_client.py`（`--mllm qwen25` 历史复算客户端）、`scripts/train_lora_g4.py`（LoRA 基座）。主线（`--mllm mock`/`qwen`、批次生成、审核工作台、转换、评测）**一处都不读**：`Qwen3VLClient` 有自己的 `_ensure_loaded`，既不调用父类那份、源码里也不出现 `QWEN_MODEL_PATH`（已核对）。文档层面 README §? 与 `docs/development/agent.md` 只是写明路径，删除时需同步改。
  2. **改名**：`models/models/qwen--Qwen2.5-VL-7B-Instruct` → `.../qwen--Qwen2.5-VL-7B-Instruct.PENDING-DELETE-20261015`（16 GB，同盘 rename 瞬时完成、不释放空间）。该目录在它自己的 `.gitignore` 里，改名后那个仓库 `git status` 仍然干净。改名**不是因为确定要删**，而是为了把"谁在用"变成可观测：路径一旦悬空，真正的依赖会立刻暴露。
  3. **暴露出的失败方式很糟，顺手修掉**：`build_pipeline("qwen25")` 在权重缺失时**照样构造成功**（懒加载），直到第一次推理才由 transformers 抛出 `Repo id must be in the form 'namespace/repo_name'`——因为它把不存在的本地路径当成了 hub 仓库名。现在两条路径都在加载前明说：客户端 `_ensure_loaded` 与 `main.build_pipeline` 各加一条存在性检查，报错点名缺失目录、说明"这是历史复算用的旧模型，g8 主线用 Qwen3-VL 不受影响"，并指出去哪条路（`--mllm qwen`）。加 2 条测试。
  4. **删除会影响的另一个项目（必须报给用户）**：`psychology_video_project` **自己**把这份权重当作主模型——`scripts/run_streaming_batch.py` 与 `scripts/run_streaming_pipeline.py` 都按 `Path(__file__).parent.parent/"models"/"models"/"qwen--Qwen2.5-VL-7B-Instruct"/"snapshots"/"master"` 加载（硬编码，无环境变量可覆盖），README 也写明"Qwen2.5-VL 用于 A3 事件复核"。也就是说：删掉它，那个项目的场景语义复核阶段会跑不起来。**这一条属于用户决定，不是我能代答的。**
* **涉及/修改的文件清单**：`mllm/qwen_client.py (Modified — 缺失即明说 + 补 `import os`)`、`main.py (Modified — qwen25 分支前置检查)`、`tests/test_g8_entry.py (Modified — 新增 2 条 + 1 条既有用例适配前置检查)`、`claude_operation_log.md`；**文件系统**：`/root/autodl-tmp/psychology_video_project/models/models/qwen--Qwen2.5-VL-7B-Instruct{,.PENDING-DELETE-20261015}`（改名，未删）
* **执行结果与验证状态**：**改名后** `full` **1420 通过 / 3 跳过**（新增 2 条）；**改名后在真实 GPU 上跑通主线冒烟**：Qwen3-VL 六项检查全绿（64.9 s、峰值 19.65 GB，产物 `sft_data/g8_smoke/20261015_224341/`）——这是"前项目不依赖它"的直接证据；`--mllm qwen25` 现在报的是明确的一句话而不是 hub 报错。当前项目未改动任何批次材料/轨迹/台账。**未跑训练、未删除任何文件。**
* **置信度或遗留待办（TODO）**：①**改名不释放空间**；确认无影响后再删才释放 16 GB②删除时需同步：`config.QWEN_MODEL_PATH` 的处置（删常量还是保留）、README 与 `docs/development/agent.md` 里的路径段落、以及"g7 复算"能力说明（g7 已冻结，复算要真权重）③`train_lora_g4.py` 若还要用，得改基座或改路径④`psychology_video_project` 其余内容（321 MB `.git`、~1.8 MB 源码与文档、6.6 MB `yolov8n-pose.pt`、空的 `dataset/`）与当前项目**无任何交叉引用**，删不删是那个项目自己的事。

### 2026-10-16 - 阶段四 4.6d：人工审核结果写回台账（只做写回，未改任何协议/代码）

* **当前操作动作**：把用户上传的审核导出 `review_g8_b1.json` 写回验收台账并重建页面显示；**未改提示词、未改代码逻辑、未动 b1 记录与轨迹**。
* **对应计划锚点**：`plan.md` §4.6（审核工作台与写回路径）、§4.5（b1 小批）；用户 2026-10-16 指示"先只做 a"
* **核心变更说明**：
  1. **写回**：`python scripts/g8_review_import.py sft_data/g8/review/b1/review_g8_b1.json` → 追加 32 行到 `sft_data/g8/review/b1/dispositions.jsonl`。校验前置 `--dry-run` 一致：接受 32、过期 0、拒收 0。
  2. **内容**：审核人 hj；**通过 14 / 待修 18 / 拒绝 0**；分类 `evidence_citation` 17、`visual_grounding` 1；每条待修都填了原因；32 条指纹与当前样本完全一致（无过期、无跨批混入）。台账字段不含任何训练标记（`scope` 为验收用途）。
  3. **页面**：重建后索引与会话页显示"仓库台账"状态（32 条已写回），本地草稿与仓库台账两种状态在页面上分开。
  4. **意见主题（供下一轮修协议参考，本次不改）**：5 条＝拿"没有其他专家冲突"当已完成交叉验证；4 条＝误用 "presumption of innocence"；3 条＝画质/分辨率当生成佐证或由观感推出高分辨率；3 条＝声称已排除压缩/编码影响；2 条＝声称未发现传感器噪声；1 条＝把整图分类说成未检测到局部异常。其中"疑罪从无"与"注明压缩"两句**出自我们自己的提示词模板**（`FORENSIC_SYSTEM_PROMPT` 的 Structure B），属协议层问题而非模型自由发挥。
* **涉及/修改的文件清单**：`sft_data/g8/review/b1/dispositions.jsonl (Created — 32 行)`、`sft_data/g8/review/b1/*.html (Rebuilt — 显示台账状态)`、`claude_operation_log.md`
* **执行结果与验证状态**：写回 32/32；台账字段与唯一性核对通过（g8/b1、审核人一致、无 `trainable` 字段）；页面 `repo_decisions` 32 条、样本页显示台账审核人。**未改提示词/代码；未触发任何训练产物。**
* **置信度或遗留待办（TODO）**：①修协议的具体方案待与用户讨论后实施（涉及提示词哈希重钉，b1 保持不动作为对照）②"是否要求第二意见"是行为决定，需用户定③b1 的 18 条待修是否要按意见重跑，取决于①的结果

### 2026-10-16 - 阶段四 4.7：按审核意见修协议（提示词 A+B+C）+ confidence 语义统一

* **当前操作动作**：按用户确认的方案（范围＝A+B+第二意见声明，含三处调整与 confidence 契约）改提示词与字段语义，并审计 `confidence` 的下游消费者。**b1 记录、轨迹、台账与审核结论一律未动**（指纹三方一致已复核）。
* **对应计划锚点**：`plan.md` §4.6（工作台与回写）、§5（分数边界）；用户 2026-10-16 的三处调整与 confidence 要求
* **核心变更说明**：
  1. **A1 彻底删除"疑罪从无"**：`Structure B` 模板不再教 "apply presumption of innocence"；新模板要求"方向不一致时点名冲突专家、其读数与适用范围，再据证据决定继续/结案/弃权"，并保留一条**禁令**："Never invoke a \"presumption of innocence\" instead of analysing the evidence."（该短语现在全文只出现这一次，且是否定式；测试钉住计数＝1 与所在语境。）
  2. **A2 压缩句改成边界规则**：删掉 "If the image has compression artifacts … note it"；新规则（RULES 8）区分**现象与成因**：可以描述可见的块状/模糊，但不得断言成因，也不得声称压缩、重编码或传感器噪声"已被测量/排除/证明无影响"——只有真正测量它的专家能说，且只在其 `scope` 内；分辨率一律取输入元数据，不得由观感推断"高分辨率"。
  3. **C 第二意见声明（并入同一条规则，不新增字段/模板）**：RULES 7 要求**每条结论**说明证据范围——有几个不同专家返回了**有效测量**、是否做了**跨专家对照**；用词规定为 "cross-expert comparison"，明令不得写成 "independent cross-validation"（两个检测器可能共享训练来源或同时失效）；缺失证据带来的限制**只在相关时**说明。B3 的单专家声明与该条合并，不重复设字段。
  4. **confidence 契约（RULES 9 + 判决 schema 字段说明 + 训练基线一句）**：`confidence` 是模型对**所选结论**的主观、自报、**未校准**把握；不是 P(Real/Fake)、不是校准概率、不是正确率；不得由专家分数复制/取补/平均/相乘构造；三件事严格分开（专家原始分数、模型自报置信度、项目实测性能）；`Uncertain` 时它指"对弃权的把握"，绝不是 P(Fake)；不要求每条报告重复免责声明——语义由契约、字段定义与工作台标签统一承担。
  5. **confidence 下游审计（用户点名）**：停止策略（v2/v3/controller）**一处都没把 confidence 当概率**（判断全走 posterior/冲突/收益）；没有地方用 confidence 覆盖标签；准入/门禁里的 0.8 阈值是**标记**（"自称很确定却判错"）而非概率。**真正当概率用的只有三处**：①`utils/final_v2.py` 对 **v2 记录**的一致性等式（v2 策略把 confidence 定义为 P(标签)，属记录校验，已有 `model_led` 守卫隔离 v3，本次把隔离写进注释）；②`scripts/qwen_gain_baseline.py::_pseudo_probability`（→AUROC/ECE，被 `check_acceptance_g5.py`、`analyze_g2_gain.py` 复用于**历史臂**，本次标明"仅 v2"）；③`scripts/generate_trajectories_g4.py` 的 `model_probability`——**原先没有策略守卫**（v3 会把自报把握折成 P(Fake) 并污染 model_nll/brier），本次按策略守卫：非 v2 写 `None`。
  6. **工作台**：置信度行改为「模型自报置信度（未校准）」并写明"不是 P(Real/Fake)、也不是专家分数"；新增「本次得到的专家原始分数（未校准；非百分比）」表（expert/状态/范围/分数/该值的方向），数据**从投递原文解析**（不重算）；并加测试禁止渲染 "N% 真实/伪造" 这类写法。
  7. **顺手删掉一段死代码**：`TOOL_READ_NOTES["probe"]` 永远进不了提示词（4.4 起 probe 的说明由登记表生成，而该表在拼接时过滤登记表标签）。它唯一有用的一句（PROBE 分数不是校准概率）已搬进登记表 `prompt_note`——**那里才是模型真正读到的地方**；旧句 "cite it as a probability" 随之消失。
  8. **哈希**：`a01272f96bbd66c7 → db111b1eb1ecb03d → 434ac95a6bbb3e59`（后一次为死代码搬迁与登记表措辞），测试里写明每次变更的原因；提示词长度 8000 → 约 10.7 KB（每次调用都带，属真实成本，已如实记录）。冻结的 `BASELINE_SYSTEM_PROMPT`（G2-d/G3-e 对照臂）**未动**，并有测试钉住它没被顺手扩写。
* **涉及/修改的文件清单**：`mllm/message_builder.py (Modified — Structure B 模板、RULES 7/8/9、判决 schema 说明、训练基线一句、删死代码)`、`experts/score_semantics.json (Modified — probe 的 prompt_note 增一句)`、`utils/final_v2.py (Modified — v2 隔离注释)`、`scripts/qwen_gain_baseline.py (Modified — 标明仅 v2)`、`scripts/generate_trajectories_g4.py (Modified — model_probability 按策略守卫)`、`scripts/admit_trajectories_g4.py (Modified — 阈值语义说明)`、`scripts/g8_review_workstation.py (Modified — 两个量分别显示 + 专家分数表)`、`tests/test_message_builder.py`、`tests/test_g8_review_workstation.py`、`tests/test_generate_trajectories_g4.py`、`sft_data/g8/review/b1/*.html (Rebuilt — 仅标签)`、`claude_operation_log.md`
* **执行结果与验证状态**：`full` **1431 通过 / 3 跳过**（另有 2 条失败，**都是环境依赖**：本机 GPU 已不在——`nvidia-smi` 报 No devices found、torch `cuda.is_available()=False`，于是 CUDA 门禁抢先于权重检查、UnivFD 如实报"不能在 CPU 上跑"。两条已改成与机器无关：前者假装 CUDA 可用以测权重检查、后者显式 `device="cuda"`；改后 1433 通过）；**b1 未受影响**：台账/页面/独立复算三方指纹一致（32/32，`stale` 0，进度 32/32），审核结论未变；工作台页面已重建并抽查（两个量分别显示、专家分数表取值来自投递原文）。**未跑 GPU**——协议变更的验证需要下一批（见 TODO）。
* **环境**：本次运行期间 GPU 被释放/切换（`nvidia-smi` 无设备、torch 无 CUDA）——**下一批（b2）需要用户重新开启 GPU 才能跑**；协议修订本身已可在 CPU 上测。
* **置信度或遗留待办（TODO）**：①**这三处修订要下一批（b2）才被验证**：提示词改了但模型行为是否跟着改，只能实跑看②`admit_trajectories_g4.py` 的 0.8 阈值、`audit_sft_correct.py` 的 0.95——按新语义它们是"自称确定"的标记而非概率，**保留**；若要改成别的量（例如只在多专家会话上标记），需用户定③顺带发现的小 bug 未改：`halting_v2.py` 无工具分支用 `confidence or 0.5`，会把合法的 0.0 当成缺失 → 0.5；改它会影响 v2 复算的既有数字，故只记录不动④b1 的 18 条待修是否重跑/是否作为对照保留，等下一批方案定

### 2026-10-16 - 外部评测数据下载（plan §9）

* **当前操作动作**：按用户“全部下载”授权，在本机下载 MNW 四个目录及 DIV2K Validation HR，不运行模型或GPU。
* **对应计划锚点**：`plan.md` §9。
* **核心变更说明**：MNW固定 revision `42b0188b580bc955d54b26f2ea7016021fc708e7`，四个生成器各250张；DIV2K 100张；原始图片不重编码。保存完整来源/大小/SHA256清单；图片与ZIP加入忽略规则，保留可续跑脚本。
* **涉及/修改的文件清单**：`scripts/download_mnw_div2k.py`、`.gitignore`、`plan.md`、`README.md`、`dataset/mnw_div2k/{README.md,download_inventory.json,manifest.json}`；本地图像与ZIP在该目录内、不入Git。
* **执行结果与验证状态**：1100张图片、1,148,911,774字节；1000张MNW匹配官方LFS大小与SHA256，DIV2K ZIP CRC通过；全部图片通过Pillow完整性检查；独立重算1100文件大小/哈希一致、无重复SHA256、无残余part；脚本语法检查通过。没有修改已有源图、分区、轨迹或审核台账。
* **置信度或遗留待办（TODO）**：尚未抽样/评测。组合存在类别不平衡与格式/分辨率/内容差异；不得声称绝对未见。MNW只作非商业评测、不用于SFT，DIV2K限学术研究。ZIP保留用于复验，未删除用户数据、未推送远端。

### 2026-10-16 - 工程规范资产描述同步（plan §9.1）

* **当前操作动作**：按用户要求更新 `docs/development/agent.md` 的大模型和数据集资产描述。
* **对应计划锚点**：`plan.md` §9.1。
* **核心变更说明**：将主模型说明改为Qwen3-VL-8B固定revision/BF16/non-thinking/transformers，列出AutoDL实测快照与显存；旧Qwen2.5单列历史依赖与改名未删除状态。新增MNW/DIV2K路径、数量、哈希清单和评测用途，修正 `/dataset` 为项目相对路径，明确本机下载不代表已上传AutoDL。
* **涉及/修改的文件清单**：`docs/development/agent.md`、`plan.md`、`claude_operation_log.md`。
* **执行结果与验证状态**：对照 `mllm/qwen3_client.py`、`main.py`、冒烟报告和外部数据manifest核对字段；仅文档更新，无模型加载、GPU或数据移动。
* **置信度或遗留待办（TODO）**：AutoDL旧权重状态来自已有日志，未重新远程确认；新版训练入口的旧基座问题仅如实记录，本轮不修改代码。

### 2026-10-16 - 提交前资产状态更正（plan §9.2）

* **当前操作动作**：按用户要求同步现存资产表述并本地提交。
* **对应计划锚点**：`plan.md` §9.2。
* **核心变更说明**：1100张图片仍在；此前清单、目录说明及ZIP已不在目录内。历史下载核验记录保留，当前文档明确区分当时验证与现存材料；不补造清单，不重复下载，不删除图片。
* **涉及/修改的文件清单**：`.gitignore`、`README.md`、`docs/development/agent.md`、`plan.md`、`claude_operation_log.md`、`scripts/download_mnw_div2k.py`（连同此前未提交的下载准备与资产同步改动）。
* **执行结果与验证状态**：提交前检查五组数量、图片完整性、本地SHA256去重及脚本语法；检查diff和暂存范围，禁止图片/权重/ZIP入库。没有运行模型或GPU。
* **置信度或遗留待办（TODO）**：现有文件哈希可本地重算，但缺少此前manifest，本轮不能重复声称已与官方哈希逐项比对。未上传AutoDL、未评测；仅本地提交，不推送远端。

### 2026-10-17 - 阶段四 4.8：外部数据清单入库（只提交清单，图片不入库）

* **当前操作动作**：按用户指示把 `dataset/mnw_div2k/manifest.json` 纳入 git；提交前做只读核对，并在 plan §9.3 写明**用途限制**。**未提交任何图片、未改代码、未跑模型/GPU。**
* **对应计划锚点**：`plan.md` §9（外部评测数据准备）、§9.2（"后续评测前另行固定样本及其哈希清单"）、§9.3（本次新增）
* **核心变更说明**：
  1. **只暂存清单一个文件**：`git add dataset/mnw_div2k/manifest.json`，暂存区仅此一项；图片目录继续由 `.gitignore`（`Fake/`、`Real/`、`downloads/`）忽略，1.1 GB 不会误入仓库。
  2. **只读核对结果**：清单 1100 条（Adobe_Firefly_v4 / ByteDance_Seedream_v4 / Google_Imagen4 / Midjourney_v7 各 250＋DIV2K_valid_HR 100）；**清单↔磁盘双向一一对应**；随机 40 张 sha256＋字节数 **40/40** 一致；清单内**无重复 SHA256**；`images_reencoded: false`。
  3. **用途限制写进 plan**：`purpose = External evaluation only; do not use for SFT or threshold fitting`；MNW 仅评测/非商业，DIV2K 仅学术；caveats（非配对测试集、不保证与骨干训练数据逐图不相交、格式/分辨率/内容混杂需分开分析）一并保留。**明确记录：清单解锁的是外部评测，不解锁训练。**
* **涉及/修改的文件清单**：`dataset/mnw_div2k/manifest.json (Added — 1100 条清单)`、`plan.md (Modified — §9.3)`、`claude_operation_log.md`
* **执行结果与验证状态**：暂存与提交范围核对（1 个文件、8818 行）；清单核对通过（见上）；仓库内无图片/权重/ZIP 被加入。**未推送远端**（按项目规矩只本地提交）。
* **置信度或遗留待办（TODO）**：①外部评测开跑前仍需：GPU、固定抽样清单（从 1100 中按格式/来源分层抽样）、报告口径（格式/分辨率/内容混杂分开）②训练线不受此清单影响，仍待 b2 验收与协议验证③清单只保证"与本机文件一致"，不重述当时与官方 LFS 的逐项比对（那份旧清单已不在；本清单自带 url 与 sha256，可在需要时重新核验）

### 2026-10-18 - 阶段四 4.9：RULES 5 不再按证据条数强制结案；b2 与外部评测预注册

* **当前操作动作**：按用户指出的设计冲突改写提示词 RULES 5（"收到 2+ 证据后必须输出结论"）；把 **b2 预注册**与**外部评测方案**写进 plan 并生成固定抽样清单；提交只读审计脚本。**未跑任何模型/GPU，未动 b1 数据与台账。**
* **对应计划锚点**：`plan.md` §4.8（b2 预注册）、§4.9（外部评测方案）；用户 2026-10-18 的两次确认（同 8 源配对；4 生成器各 20 + Real 20）
* **核心变更说明**：
  1. **RULES 5 改写**：旧文 "After receiving 2+ evidence tokens, you MUST produce a verdict"（引入于 `fac5c36`，G1/v2 时代；**代码里从未强制执行**，只是提示词）。它与 v3"模型决定何时结案"的前提冲突——在第 2 条证据后剥夺判断权，恰是两份测量冲突、再来一个专家可能有用的时刻；与 4.7 的 RULES 7（据证据决定继续/结案/弃权）语义打架；还制造反向激励（想继续查就别打第二个电话）。新文：**充分性由模型判断**（"Conclude once the measurements you called for have arrived and another call would not change your reading"），**兜底交给系统预算**（"the system's budget — not a count of tokens — is what ends a session"），弃权原则原样保留。
  2. **b2 预注册（§4.8）**：`g8 / b2`，与 b1 同 8 源 × 4 格式（同 seed，配对），32 会话；除提示词外条件全同（Qwen3-VL@`0c351dd0`、v3、精简包、probe/npr/univfd、贪心、非 thinking）。判据先写死并带 **b1 实测基线**：presumption 4 处/4 会话→**0**；overclaim 19 处/18 会话→**≤6 会话**；excluded 23 处/16 会话→**≤6 会话**；结案声明证据范围 **0/32→32/32**；判定一致率不降；引用检查 0 缺陷、全 slim。另定人工抽查 8 条防"机械加一句"。提示词指纹：b1 `af7003e17e0d6a42` → b2 `8d664a768bb74e9d`。
  3. **外部评测（§4.9）**：MNW 四生成器各 20 + DIV2K 20 = **100 张**；从已入库 manifest 按组内 sha256 排序后 `random.Random(20261018).sample` 抽取（可复现、与下载顺序无关），**不重编码**；只评三个专家的分数、不跑 Agent、**不调阈值**；报告按组单独给（n=20，判 Fake 比例、分数分布、按登记表既有阈值判对数、组内 AUROC+95%CI），格式/分辨率分别列、不跨组汇总。
  4. **固定抽样清单**：`dataset/mnw_div2k/eval_plan.json`（`external_eval_plan/1`，含源清单 sha256、抽样规则与种子、100 条 path/sha256/bytes/label）。只读核对：抽样文件全部存在、sha256 与字节数 **0 不符**、组内 **0 重复**。
  5. **审计工具**：`scripts/g8_protocol_audit.py`（只读）数四组判据并给样例；b1 基线即由它数出（presumption 4/4、overclaim 19/18、excluded 23/16、scope 0/32）。
* **涉及/修改的文件清单**：`mllm/message_builder.py (Modified — RULES 5)`、`tests/test_message_builder.py (Modified — 新契约测试 + 哈希重钉 259069a1ddb66348)`、`plan.md (Modified — §4.8/§4.9)`、`dataset/mnw_div2k/eval_plan.json (Added)`、`scripts/g8_protocol_audit.py (Added)`、`claude_operation_log.md`
* **执行结果与验证状态**：提示词哈希 `434ac95a6bbb3e59 → 259069a1ddb66348`（`prompt_digest` `8d664a768bb74e9d`）；`test_message_builder`/`test_toolbox`/`test_g8_smoke` **全部通过**；eval_plan 100 张核对通过（0 哈希不符、0 重复）；b1 数据、台账、审阅页均未改动。
* **置信度或遗留待办（TODO）**：①b2 与外部评测都需要 **GPU**（当前 `nvidia-smi` 无设备）②b2 的判据阈值（≤6 会话）已写进 plan；若实测落在 7–12 之间属"改善但不达标"，届时按数据决定是否再改协议，**不改判据**③外部评测只做专家级；若之后要看 Agent 在外部分布上的表现，另立小批（需用户定）④审计脚本只数"有没有说"；"说得对不对"仍靠人工抽查

### 2026-10-18 - 阶段四 4.9b：b2 输入逐文件相同（运行前断言）；外部评测口径与阶段修正

* **当前操作动作**：按用户三条更正调整：(1) b2 的输入文件改为 b1 的**逐字节副本**并加运行前断言；(2) 外部评测把"单组 AUROC"改为**每组 20 Fake + 20 Real**（单标签组算不了 AUROC）；(3) 外部评测改为**专家基线在前、Agent 评测在后**。**未跑模型/GPU。**
* **对应计划锚点**：`plan.md` §4.8（b2 预注册）、§4.9（外部评测方案）；用户 2026-10-18 的三条更正
* **核心变更说明**：
  1. **b2 输入逐文件相同**：`sft_data/g8/variants/b2/` 不是"用同 seed 重建"，而是 **b1 的 32 个文件逐字节复制**；b2 清单里每条记 `sha256`、`bytes`、`copied_from`，并记 b1 清单的 sha256。核对：文件名集合相同、**32/32 sha256 相同**。
  2. **运行前断言**：`g8_generate_batch.py` 新增 `--assert-variants-identical-to <对照批>`，开跑前对**真实文件**计算 sha256 并逐条比对，任何一处不同/缺失即 `SystemExit` 拒绝启动（"同 seed 重建"只是假定编码器确定性，名字来源相同而字节不同在记录里看不出来）。加 2 条测试（相同放行 / 字节不同拒绝）。
  3. **外部评测口径修正**：原先写"每组单独报 AUROC"是错的——每组只有一种标签。现改为：**每组 = 某生成器 20 张 Fake ＋ 同一组 20 张 DIV2K Real（n=40）**，这样组内 AUROC 才可算；**20 张 DIV2K 在四个生成器组间复用**，四组因此不独立，报告须写明；仍不跨组汇总、不合并成单一数字。指标：组内 AUROC+95%CI、按登记表既有阈值判对的数、Fake/Real 两组分数分布。
  4. **两阶段外部评测**：阶段 1 专家基线（PROBE/NPR/UnivFD，不调阈值）；阶段 2 在专家基线之后跑 Agent（固定子集 10/组 × 5 组 = 50 会话，`Random(20261019)` 抽样、已写入 `eval_plan.json`），按 §4.8 口径记录结案声明，仍属评测、不用于 SFT。
* **涉及/修改的文件清单**：`sft_data/g8/variants/b2/`（32 个字节副本 + `manifest.json`）、`scripts/g8_generate_batch.py (Modified — 断言)`、`tests/test_g8_batch_entry.py (Modified — 2 条)`、`plan.md (Modified — §4.8/§4.9)`、`dataset/mnw_div2k/eval_plan.json (Modified — 分析配对 + Agent 子集)`、`claude_operation_log.md`
* **执行结果与验证状态**：b2 输入 32/32 sha256 与 b1 相同；`test_g8_batch_entry` **14 通过**；eval_plan 分析组 4×40、Agent 子集 50（10/组×5）。b1 数据/台账未动。
* **置信度或遗留待办（TODO）**：①b2 开跑命令将带 `--assert-variants-identical-to b1`，不通过就不跑②外部评测仍需 GPU；Agent 子集的规模（50）如需调整在开跑前改 `eval_plan.json` 并记录③DIV2K 复用导致的组间相关性只能靠"报告写明"处理，不做跨组统计检验

### 2026-10-18 - 阶段四 4.9c：工作区不干净的根因与兜底（测试夹具写进了仓库）

* **当前操作动作**：用户指出"git 工作区不干净"。查明是我新加的 b2 输入比对测试用 `batch_paths()` 拼夹具路径，把 `b_new`/`b_ref` 两个目录写进了 `sft_data/g8/variants/`。已删除该残留、把被测函数改为接收显式路径、测试改为纯 `tmp_path`；并在 `tests/conftest.py` 增加会话级兜底。**未动任何批次数据与台账。**
* **对应计划锚点**：`plan.md` §4.8/§4.9（前一条 4.9b）；用户 2026-10-18 的指出
* **核心变更说明**：
  1. **根因**：`assert_variants_identical` 内部用 `batch_paths(source_batch, generation)` 拼路径，测试因此只能建仓库内目录。改为**函数接收两个显式清单路径**（调用方算路径），测试夹具全部落在 `tmp_path`，并在夹具里断言"路径确实在临时目录下"防止回归。
  2. **兜底**：`tests/conftest.py` 新增 `tests_leave_no_untracked_files`（session 级 autouse）——会话开始/结束对 `sft_data/`、`traces/`、`dataset/` 的**未跟踪**文件做快照，结束时若多出任何路径即以 `pytest.UsageError` 让整场失败并列出文件。只比较未跟踪集合，不影响用户自己的改动。用一个一次性探针验证过：写入 `sft_data/` 的测试会让 pytest 退出码为 1。
  3. **过程中我犯的错（如实记录）**：加兜底时我用 heredoc **覆盖了已存在的 `tests/conftest.py`**（原有 169 行，含 `isolated_sft_dir`、`legacy_calibration` 等共享 fixture）。发现后立即 `git checkout -- tests/conftest.py` 恢复，再**追加**兜底 fixture；当前 diff 是 `+52/-0`（另加一行 `import subprocess`），两个关键 fixture 完好。
* **涉及/修改的文件清单**：`scripts/g8_generate_batch.py (Modified — 显式路径 + 残留变量修复)`、`tests/test_g8_batch_entry.py (Modified — 夹具改 tmp_path)`、`tests/conftest.py (Modified — 仅追加兜底 fixture + import)`、`claude_operation_log.md`；**已删除**：`sft_data/g8/variants/{b_new,b_ref}/`（我几分钟前生成的测试夹具）
* **执行结果与验证状态**：`git status` 只剩上述三个受控修改，无未跟踪产物；`core` **601 通过**（兜底 fixture 生效且未误报）；兜底探针验证退出码 1、文件被点名；`tests/conftest.py` 原始 fixture 完整（`isolated_sft_dir`/`legacy_calibration` 均在）。
* **置信度或遗留待办（TODO）**：①兜底只拦"未跟踪的新文件"，测试若修改已跟踪文件不会被发现（当前无此类测试；如有需要再加内容比对）②`_untracked_under` 依赖 `git status`，在非 git 环境会得到空集合（不误报、也不保护）③之后新增测试请一律用 `tmp_path`

### 2026-10-18 - 阶段四 4.10：两条任务线定稿；外部 200 输入与批次身份补全

* **当前操作动作**：按用户 2026-10-18 的完整方案整理两条任务线（b2 协议验收 / 外部泛化评测），补齐"固定模型/权重/参数/预算/代码"的身份记录，生成外部 200 个输入文件（100 源 × original/jpeg_q95），把协议命中接进审阅页，审计脚本加批次汇总。**未跑任何模型/GPU；b1 数据与台账未动。**
* **对应计划锚点**：`plan.md` §4.8（b2 预注册）、§4.9（外部评测，本次重写）；用户 2026-10-18 的完整方案
* **核心变更说明**：
  1. **两条线不产生训练数据**：批次元数据新增 `purpose`（`protocol_acceptance_only` / `external_evaluation_only`）；b2 与外部评测都不跑 admission/conversion。
  2. **批次身份补全**（用户"固定模型 revision/专家权重/预处理/推理参数/安全预算，记录实际代码和提示词身份"）：`metadata` 新增 `code`（git commit、是否 dirty 及 dirty 路径、分支）、`budget`（停机策略、MAX_EXPERT_CALLS、MAX_MODEL_TURNS、投递形状）、`prompts.forensic_sha256`（会话实际使用那份提示词的哈希与字符数）；专家身份并入 `weights_sha256`（优先 `identity()`，否则对 weights_path 取哈希）与设备。**顺带修掉一个真实副作用**：mock 专家的 weights_path 是 `/dev/null`，对设备文件取哈希会在 `/dev` 下写 sidecar——`_weights_digest` 与 `ExternalScoreExpert.provenance()` 都改成只对普通文件取哈希。
  3. **外部 200 输入**（`scripts/build_external_inputs.py` → `dataset/mnw_div2k/inputs_manifest.json`）：100 源 × 2 条件。`original` 原样引用（不动原文件）；`jpeg_q95` 真假同一编码器同一参数（Pillow 11.0.0 / libjpeg 6.2 / q95 / subsampling 2 / optimize、progressive 关），**不缩放**，编码器身份与参数逐条入清单。核对：200 条、唯一 id、无重复哈希、原图哈希与固定清单一致、每个源两条件齐全、JPEG 尺寸未变；副本目录 `dataset/mnw_div2k/inputs/` 已加入忽略规则（70 MB 不入库），清单入库。`eval_plan.json` 记录 `usage_history`（何时、何用途、多少输入、编码器）。
  4. **分析口径（按用户更正）**：每组 = 某生成器 20 Fake + 同一组 20 DIV2K Real（n=40）；两种条件**同源配对**（配对 ΔAUROC，bootstrap 按源重采样）；20 张 Real 跨四组复用、组间不独立；不跨组汇总、不给单一总分。**Agent 与专家用完全相同的 200 个输入**：专家 3×200 = 600 次测量；Agent 200 会话（50 张子集已取消），专家基线先跑、Agent 在 b2 验收冻结后跑，离线分数不预注入。
  5. **审阅工作流**：审阅页新增「协议命中（自动候选）」区——逐条列出判据、轮次与**原句**，并给出"结案是否声明证据范围"的自动判断；页面明写"自动命中只是候选，未命中不等于没问题"，配合"人工复核全部命中/错误/弃权 + 抽查 8 条未命中"。
  6. **批次汇总**：`g8_protocol_audit.py` 现在同时输出**正确/错误/弃权/未形成结论**（弃权单列）、工具调用分布、停止原因、耗时（均/最长）、日志完整性（系统提示词、调用明细、图像尺寸）以及四组判据的命中率与样例。b1 基线：`system_prompt 0/32、model_calls 0/32`（4.6 之前无此记录），与 4.6 起的批次形成对照。
* **涉及/修改的文件清单**：`scripts/g8_generate_batch.py (Modified — code/budget/purpose/weights + 设备文件修复)`、`experts/external_base.py (Modified — provenance 只对普通文件取哈希)`、`scripts/build_external_inputs.py (Created)`、`dataset/mnw_div2k/inputs_manifest.json (Created — 200 条)`、`dataset/mnw_div2k/eval_plan.json (Modified — agent_plan/配对/usage_history)`、`scripts/g8_protocol_audit.py (Modified — 批次汇总)`、`scripts/g8_review_workstation.py (Modified — 协议命中区)`、`.gitignore (Modified — inputs/ 忽略)`、`plan.md (Modified — §4.9 重写)`、`tests/{test_external_inputs.py (Created), test_g8_batch_entry.py, test_g8_review_workstation.py, test_external_experts.py}`、`sft_data/g8/review/b1/*.html (Rebuilt — 仅新增协议命中区)`、`claude_operation_log.md`
* **执行结果与验证状态**：`core` **611 通过**；外部输入 200 条核对通过（唯一 id、无重复哈希、原图未动、双条件齐全、无缩放、编码器身份一致）；b1 页面重建后台账进度仍 32/32、`stale` 0（指纹未变）；`/dev/null.sha256` 副作用已消除。**未跑 GPU、未产生任何训练数据。**
* **置信度或遗留待办（TODO）**：①b2 与外部评测都需要 GPU（当前无设备）②b2 开跑命令将带 `--assert-variants-identical-to b1 --purpose protocol_acceptance_only`；外部 Agent 批次将带 `--purpose external_evaluation_only`③外部专家基线尚未有专用入口脚本（三个 evaluator 已有 `--device cuda`，需按 200 输入清单写一个批量入口；下一步实现）④报告模板（按组×条件的配对 ΔAUROC）待实现⑤"Agent 200 会话"的审阅量需按 4.8 的流程走

### 2026-10-18 - 阶段四 4.10b：外部输入清单的路径基准统一（提交后自查发现）

* **当前操作动作**：发现 `inputs_manifest.json` 中 `original` 条目的 `path` 是相对 `dataset/mnw_div2k/`，而 `jpeg_q95` 条目是相对仓库根——批次入口按 `PROJECT_ROOT + path` 打开文件，前者会解析到错误位置。统一为**相对仓库根**，重建清单并加测试。**未跑模型/GPU。**
* **对应计划锚点**：`plan.md` §4.9（外部评测输入）；同日前一条 4.10
* **核心变更说明**：`build_external_inputs.py` 里 `original` 的 `path` 改为 `os.path.relpath(source, PROJECT_ROOT)`；`source_path` 仍保留相对 `dataset/mnw_div2k` 的原值（用于与固定清单配对）。重跑构建：**只有 100 条 `original.path` 变化，所有 sha256 不变**（也证明 JPEG 重编码是确定性的）；200 条路径全部可按 `PROJECT_ROOT` 解析；`usage_history` 记为 2 条。
* **涉及/修改的文件清单**：`scripts/build_external_inputs.py`、`dataset/mnw_div2k/inputs_manifest.json`、`dataset/mnw_div2k/eval_plan.json (usage_history)`、`tests/test_external_inputs.py (Modified — 路径解析回归测试)`、`claude_operation_log.md`
* **执行结果与验证状态**：`test_external_inputs` 全部通过；清单 diff 仅 100 条 path（+100/−100），无哈希变化。
* **置信度或遗留待办（TODO）**：①外部专家基线的批量入口脚本仍未写（下一步）②GPU 仍无设备

### 2026-10-18 - 阶段四 4.11：外部专家基线的评测入口与统计（CPU 可测）

* **当前操作动作**：新增 `scripts/g8_external_eval.py`——按 `inputs_manifest.json` 的 200 个输入逐条跑三个专家（与 Agent 同一套专家类/预处理），写 `scores.jsonl`（可 `--resume`）与 `report.json`；统计全部为纯函数并有测试。**未跑 GPU/模型。**
* **对应计划锚点**：`plan.md` §4.9（外部泛化评测）
* **核心变更说明**：
  1. **统计口径按用户更正**：`auroc()` 为 Mann-Whitney 形式（并列取平均秩），**单标签组返回 None**（组内单标签就是算不了，不许猜）；`bootstrap_auroc()` 按条目重采样、定种子；`paired_delta()` 对同一批源在两种条件下的 ΔAUROC 做**成对** bootstrap。阈值判对率用**登记表既有阈值**，`higher_is_real` 的专家在定向与判假两处都翻转，不拟合、不调参。
  2. **报告**：按 `group × condition` 输出 n / n_fake / n_real、每个专家的 AUROC+95%CI、阈值判对率，并把 `paired_delta_vs_jpeg_q95` 挂在 `original` 条件上；报告头写 `no_threshold_fitting / per_group_only / real_images_reused_across_generators / paired_conditions` 与清单 sha256。
  3. **入口行为**：`--dry-run` 只列分组计数（已核对：每个专家 200 次，40 组 = 4 生成器 × 2 条件 × 20 Fake + 20 Real/组口径一致）；`--resume` 按 (variant_id, expert) 跳过已完成；逐条写盘并 flush，中断不丢；跑完把使用历史追加到 `eval_plan.json`。
  4. **测试**（`tests/test_external_eval.py`，15 条，core）：AUROC 完美/反转/并列/单类别；bootstrap 定种子可复现且 CI 包住点估计；配对 delta 为正且"打乱配对比率会变"；登记表方向与阈值（含 higher_is_real 的翻转）；报告的 per-condition 计数、配对 delta 挂载、用途声明、单标签组无 AUROC。
* **涉及/修改的文件清单**：`scripts/g8_external_eval.py (Created)`、`tests/test_external_eval.py (Created)`、`plan.md (Modified — §4.9 入口与报告口径)`、`claude_operation_log.md`
* **执行结果与验证状态**：`test_external_eval` **15 通过**；`--dry-run` 在 200 输入上给出每个专家 200 次测量的分组计数；**未跑真实专家（无 GPU）**。
* **置信度或遗留待办（TODO）**：①真实运行需 GPU（UnivFD 必需）与 `PROBE_WEIGHTS`②报告尚未生成 markdown 摘要（JSON 为主，必要时再加）③外部 Agent 200 会话的入口复用 `g8_generate_batch.py --purpose external_evaluation_only`（清单路径已修）④GPU 仍无设备

### 2026-10-18 - 阶段四 4.11b：外部清单直接接入批次入口（避免 GPU 时才暴露）

* **当前操作动作**：外部 Agent 评测复用同一条批次管线时，`inputs_manifest.json` 的键是 `inputs`（列表）而批次入口只认 `variants`（字典），且条目缺 `source_id/split/generator/treatment/container`——真跑时会在第一张图前报错。已补齐并加测试。**未跑 GPU。**
* **对应计划锚点**：`plan.md` §4.9（外部评测；Agent 与专家同输入）
* **核心变更说明**：`build_external_inputs.py` 为每条输入补 `source_id`（`<组>/<stem>`）、`split="external"`、`generator=<组名>`、`treatment=original|jpeg_q95`、`container`（原图按扩展名、JPEG 条件固定 `jpg`）；`g8_generate_batch.load_manifest` 同时接受 `variants`（字典，项目内变体）与 `inputs`（列表，外部输入），两者归一为"路径→条目"。外部 Agent 批次因此**复用同一条管线**（同一套记录、指纹、日志完整性），而不是另写一份等价物。
* **涉及/修改的文件清单**：`scripts/build_external_inputs.py`、`scripts/g8_generate_batch.py`、`dataset/mnw_div2k/inputs_manifest.json (Rebuilt — 仅新增字段)`、`tests/test_external_inputs.py (Modified — 入口兼容测试)`、`claude_operation_log.md`
* **执行结果与验证状态**：`load_manifest` 读出 200 条、字段齐备、`split=external` 且 `treatment ∈ {original, jpeg_q95}`；`test_external_inputs`/`test_g8_batch_entry` 通过；清单重建未改任何哈希（确定性编码）。
* **置信度或遗留待办（TODO）**：①GPU 仍无设备②外部 Agent 批次名拟定 `ext_mnw_div2k`（待用户确认）③报告模板已随专家基线实现

### 2026-10-18 - 阶段四 4.12：执行顺序定稿，外部 Agent 的"验收+冻结"前置做成硬门

* **当前操作动作**：按用户指出的风险（"GPU 一开就按①→②→③跑"会绕过第③步的人工验收与冻结前提）把顺序写进 plan §4.12，并把第 5 步的前置条件实现为**机械门禁**：`assert_protocol_accepted()`。**未跑模型/GPU。**
* **对应计划锚点**：`plan.md` §4.12（新增）、§4.8（b2 预注册）、§4.9（外部评测）
* **核心变更说明**：
  1. **顺序定稿**：①检查+真实冒烟 → ②b2 32 会话 + b1/b2 对照 + 审阅页 → ③**暂停等人工验收（不自动继续）** → ④外部专家基线可独立先跑 → ⑤b2 验收通过且协议冻结后才跑外部 Agent 200 会话。
  2. **硬门禁**：`utils/batch.py::assert_protocol_accepted()` 要求两件事同时成立——(a) `(g8, b2)` 在 `FROZEN.json` 的 `frozen` 列表里（冻结是用户显式动作，代码不自动冻结）；(b) b2 验收台账里每条样本的**最新**决定都是 `pass` 且无 `stale`。缺任一即 `SystemExit` 并说明缺什么；`g8_generate_batch --purpose external_evaluation_only` 在加载任何权重之前调用它。
  3. **不影响**：冒烟、b2 本身、外部专家基线（用户明确"专家基线可以独立先跑"）都不走这道门。
* **涉及/修改的文件清单**：`utils/batch.py (Modified — assert_protocol_accepted)`、`scripts/g8_generate_batch.py (Modified — purpose 门禁 + --protocol-batch)`、`tests/test_batch_gate.py (Created — 7 条)`、`plan.md (Modified — §4.12)`、`claude_operation_log.md`
* **执行结果与验证状态**：未冻结时实跑外部用途**被拒**（退出码 1，提示"g8/b2 还没有冻结"与正确顺序）；门禁 7 条测试通过（未冻结/全通过/有 revise/有 stale/台账不全/入口确实调用/其他用途不触发）；**门禁在创建目录与加载模型之前生效**。
* **置信度或遗留待办（TODO）**：①b2 冻结仍由用户执行（写 FROZEN.json），本阶段不代冻②外部 Agent 批次名待定（暂拟 `ext_mnw_div2k`）③GPU 仍无设备

### 2026-10-18 - 阶段四 4.13：b2 实跑完成（27 正确 / 5 弃权 / 0 错误），判据口径更正

* **当前操作动作**：GPU 开启后按定稿顺序执行第①步（检查+真实冒烟）与第②步（b2 32 会话 + 审阅页 + b1/b2 对照），随后**停下等人工验收**，未跑外部 Agent。过程中修正了账本归因与审计判据的两处口径错误。
* **对应计划锚点**：`plan.md` §4.12（执行顺序）、§4.8（b2 预注册）、§4.8b（本次结果与口径更正）
* **核心变更说明**：
  1. **第①步**：GPU/权重/输入检查通过（RTX 4090；PROBE/NPR/UnivFD 权重齐全且 `unavailable_reason` 为空；b2 的 32 个输入与 b1 **32/32 sha256 相同**）；真实冒烟六项全绿（58.9 s，峰值 19.84 GB）。
  2. **第②步（b2）**：32 会话全部 `model_concluded`、每会话 1 次工具调用；**正确 27 / 错误 0 / 弃权 5 / 未形成结论 0**；均 26.2 s；批次元数据记录 prompt `259069a1ddb66348`、digest `8d664a768bb74e9d`、code commit `e5b1707`、purpose `protocol_acceptance_only`；日志完整性 32/32（系统提示词、调用明细）。
  3. **判据结果**：presumption **4→0**；overclaim **19→0 会话**；excluded **2→0**；结论轮声明证据范围 **0/32→32/32**。四项预注册目标全部达成。
  4. **未达标项（如实）**：预注册的"判定一致率不降"未达成（32→27）。5 条弃权全部是 Fake 且 probe ≥0.98（0.980–0.9998），理由均为"仅一个专家、未做跨专家对照"。按用户口径弃权单列不计错误；是否接受为合理弃权由人工验收决定。
  5. **账本归因修正（b2 记录里 1 条 `failed` 的根因）**：模型按 RULES 8 写"视觉观察中未发现压缩块、噪点或边缘异常"，账本把"压缩"当成 jpeg 专家名、又把非断言子句的主语带进下一个断言子句，判成"声称 jpeg 专家的结果"并触发一轮纠正。修法：①非断言子句不更新主语；②物证词（压缩/噪声/频域）单独出现不算点名专家（工具原名与"××专家"仍算）。重审该会话：0 缺陷 0 候选；"真的声称未执行专家"仍会被抓（有测试）。**b2 记录保留原样，不回溯改写**。
  6. **审计判据口径更正**：初版把"无其他专家提供交叉验证"（诚实声明）与"未发现压缩伪影"（允许的观察）算成问题，导致 b1 的 excluded 被高估。现已按意图判定并加正反例测试；预注册目标不变，b1 基线在 plan §4.8b 中更新并说明。
  7. **代码身份口径修正**：`dirty` 改为只看**已跟踪文件**，且代码身份在**开跑前**采集——b2 首次记录把自己的输出目录算进了 dirty_paths。
* **涉及/修改的文件清单**：`utils/evidence_ledger.py (Modified — 归因两处)`、`scripts/g8_protocol_audit.py (Modified — 判据按意图 + 结论轮 scope 同判定 + 批次汇总)`、`scripts/g8_generate_batch.py (Modified — code identity 口径)`、`scripts/g8_review_workstation.py (Modified — 协议命中区)`、`tests/{test_evidence_ledger.py, test_g8_review_workstation.py, test_g8_batch_entry.py, test_external_experts.py}`、`plan.md (Modified — §4.8b)`、**产物**：`sft_data/g8/trajectories/b2/`（32 记录 + metadata）、`traces/sft_sessions/g8/b2/`（32 trace）、`sft_data/g8/review/b2/`（索引 + 32 页 + `protocol_audit.md`）、`sft_data/g8_smoke/20261018_200327/`、`claude_operation_log.md`
* **执行结果与验证状态**：b2 四项判据全部达成；0 错误；审阅页已生成（会自动带上协议命中句）；审计对照写入 `sft_data/g8/review/b2/protocol_audit.md`；`core` 见提交时结果；**外部 Agent 未跑**（门禁仍要求 b2 冻结+验收全通过）。
* **置信度或遗留待办（TODO）**：①**等人工验收**（全部命中/错误/弃权逐条 + 8 条抽查）②用户决定是否冻结 b2；冻结后外部 Agent 门禁才会放行③5 条弃权是否可接受、若不可接受如何修订 RULES 7 的措辞，属下一轮协议变更④外部专家基线可独立先跑（本阶段未跑）

### 2026-10-18 - 阶段四 4.14：外部专家基线实跑（600 次）与报告配对口径修正

* **当前操作动作**：按用户指令（"外部专家基线开始，也可以用 gpu，开着的。agent 先不跑。"）在 CUDA 上跑完 PROBE / NPR-A / UnivFD × 200 输入 = **600 次测量**；随后修正报告分组口径并从既有分数重建报告（**未重测、未跑 Agent**）。
* **对应计划锚点**：`plan.md` §4.9（外部泛化评测）、§4.9b（本次结果）、§4.12 第④步
* **核心变更说明**：
  1. **实跑**：`FORENSIC_ALLOW_GPU=1 PROBE_WEIGHTS=... python scripts/g8_external_eval.py --device cuda`，逐条写 `sft_data/g8_external_eval/20261018_202449/scores.jsonl`（600 行，sha256 `a4937e11…166a58bf`）；`eval_plan.json` 追加一条 `expert_baseline` 使用历史。
  2. **报告 bug（跑后发现）**：第一版 `build_report` 按"来源组 × 条件"分组，而生成器组只有 Fake、DIV2K 组只有 Real → 每组单标签，**AUROC 全为 None**，且阈值 acc 在单标签组上没有意义。改为两份视图：`groups`（来源组，单标签，只给均值与阈值判假率）+ `analysis`（**每组 = 生成器 20 Fake + 同一条件 20 DIV2K Real**，n=40，AUROC+CI、acc、配对 ΔAUROC）。
  3. **配对键修正**：同一源的 `variant_id` 在两种条件下不同（`__original` / `__jpeg_q95` 后缀），第一版按 variant_id 求交集得到 **0 对**。改为按 `source_id`（回退 `source_path`）配对；新写分数行附带 `source_id`。
  4. **审计信息进报告**：`composition` 记录每种条件"标签 × 容器"计数（original：Fake/jpg 80 + Real/png 20；jpeg_q95：Fake/jpg 80 + Real/jpg 20）；`experts` 记录登记表中的权重文件/哈希、预处理、阈值、方向与许可；`run` 记录 `scores_sha256`、mtime、重建说明。
  5. **`--report-only <scores.jsonl>`**：只重建报告，不重跑任何测量；本轮即用它修复报告，600 行分数未变。
  6. **异常分数核查（未当结论直接汇报）**：NPR 在 80 张假图上精确 0.0、UnivFD 多数假图 <0.05。核对两处：①NPR 适配器与官方 `test.py/validate.py` 一致（`sigmoid > 0.5`、整图不缩放、ImageNet 归一化），原始 logit 抽查为 −106…+19 的饱和值；②UnivFD 预处理与官方 `validate.py:199` 一致（CenterCrop(224) → ToTensor → CLIP Normalize，无 resize）。结论：**分数是真的，不是预处理/加载 bug**；NPR 的分数跟随容器/压缩史（DIV2K 原图 PNG 均值 0.3634、35% 判假；转 jpeg_q95 后 0.0451、5%），UnivFD 在 3/4 生成器上接近或低于随机。结果与限制写入 plan §4.9b。
* **涉及/修改的文件清单**：`scripts/g8_external_eval.py (Modified — 配对分析/--report-only/审计字段)`、`tests/test_external_eval.py (Modified — 配对口径与回归，19 条)`、`plan.md (Modified — §4.9b)`、`dataset/mnw_div2k/eval_plan.json (Modified — usage_history)`、**产物**：`sft_data/g8_external_eval/20261018_202449/{scores.jsonl, report.json}`、`claude_operation_log.md`
* **执行结果与验证状态**：600/600 测量完成；报告按 4 组 × 2 条件 × 3 专家给出 n=40 的 AUROC+95%CI 与配对 Δ；`test_external_eval` **19 通过**；分数文件 sha256 已写进报告；**未跑 Agent，未生成任何 SFT 数据，未调阈值**。
* **置信度或遗留待办（TODO）**：①报告只从分数重建，**原始 run 的 device/elapsed 字段未保留**（以本日志与 plan 为准）②b2 人工验收仍未发生，外部 Agent 门禁仍关闭③PROBE 上游训练语料未声明，"外部"只指未参与本项目选择/训练，不得写成未见生成器泛化④NPR 的格式受限结论需要在下游证据语义中落实（证据必须带格式条件与 scope）⑤是否把 `--report-only` 写进 plan §4.9 入口说明（现仅在 §4.9b 记录）
