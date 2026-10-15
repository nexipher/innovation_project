# 计划（当前代次 g8）

> **本文件从 2026-10-15 起只记当前代次的工作，只追加、不改写已写下的条目。**
> v1 计划（Qwen2.5-VL + PROBE/noise/jpeg/frequency_v2 时代，含其全部阶段与决策记录）已归档：
> `archive/legacy_v1_20261015/docs/plan_v1.md`，配套操作日志
> `archive/legacy_v1_20261015/docs/claude_operation_log_v1.md`。
> 归档索引：`archive/legacy_v1_20261015/MANIFEST.json`；旧路径映射：`archive/PATH_MAP.json`。
>
> 上位文档：`docs/design/NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md`（设计原案，有效内容见 §2–§6）、
> `docs/archive/NEW_GENERATION_NAMING.md`（代次/冻结/归档映射的规则）、
> `docs/design/CURRENT_PROGRAM_ARCHITECTURE.md`（当前架构）、
> `KNOWN_ISSUES.md`（问题清单）、`docs/development/TESTING.md`（测试分层与运行方式）。
> 文档布局见 §1b：**根目录只保留四份日常文档**，其余在 `docs/{design,development,archive}/`。

## 0. 当前状态（2026-10-15）

| 事实 | 值 |
|------|-----|
| 当前代次 | `g8`（`sft_data/g8/`，模型 Qwen3-VL-8B **待确认版本**；`g7` 及更早已冻结） |
| 默认策略 | `v3`（Qwen 主导；`config.HALTING_POLICY`，可用 `FORENSIC_HALTING_POLICY` 选择 v1/v2 复算） |
| 数据布局 | 代次是目录、批次名是纯名字：`sft_data/<generation>/{trajectories,variants,review,train/batches}/<batch>/`；三个写入入口的 `--generation` **必填** |
| 冻结 | `sft_data/FROZEN.json` 以 **(代次, 批次)** 为身份，代码层拒收；**两级**：未登记代次一律拒、冻结代次下不得新建任何批次、已登记冻结批次走 `--allow-frozen` 复算。九个写入入口同一函数 |
| 归档 | v1 全部产物已移入 `archive/legacy_v1_20261015/`（34 项、2141 个文件），逐文件核对通过 |
| 上一批实测 | `g7_b2`（56 会话，v3）：协议合规 56/56、0 违规、0 解析失败；但 **PROBE 单专家 53/55 优于 Agent 44/56** —— 这是换代的直接依据，不是待修的小问题 |

## 1. 阶段 1（**已完成，2026-10-15**）：目录隔离、冻结升级、归档

清单见 `docs/archive/NEW_GENERATION_NAMING.md` §6。执行记录（每一步做了什么、核对了什么）见
`claude_operation_log.md` 2026-10-15 的两条。当前状态：

- [x] 1. `utils/batch.py`：`batch_paths(batch, generation=None)` 支持嵌套布局；`assert_not_frozen_batch` 要求显式代次；`frozen_batches()` 返回 `{(generation, batch): entry}`；`resolve_path()` 查 `archive/PATH_MAP.json`
- [x] 2. `sft_data/FROZEN.json` 加 `generations` 索引；新建 `sft_data/GENERATIONS.json`；建 `sft_data/g8/` 空骨架
- [x] 3. 写入入口加**必填** `--generation`：`generate_trajectories_g4` / `build_final_v2` / `apply_review` 之后，逐次补齐 `build_variants_g4`、`admit_trajectories_g4`、`build_review_workstation`、`record_review`、`build_review_from_trajectories`、`g7_select_sources`——**共九个**，清单写在测试里（`tests/test_batch_binding.py` 的 `WRITERS`），漏一个就红
- [x] 3b. 冻结检查升级为**两级**（`assert_generation_is_writable`）：未登记代次一律拒（拼错代次名无 `--allow`）；冻结代次下**不得新建任何批次**（换个没登记的批次名也不行）；已登记的冻结批次仍可用 `--allow-frozen` 做带标记的复算
- [x] 4. 归档移动：docs+history 34 项 → `archive/legacy_v1_20261015/{docs,experiments,review}/`；**代码一个都没动**
- [x] 5. 写 `MANIFEST.json`（四段结构 + 留在原位清单 + 使用限制）、`PATH_MAP.json`、`NOTE.md`
- [x] 6. 移动前后摘要核对：`archive_inventory.py --verify` 通过；另加 `archive_v1.py --verify` 逐文件对 `HEAD` blob 核对（含 438 页登记过的链接改写）
- [x] 7. 路径修正：历史复算脚本走 `resolve_path` 并在报告里写 `resolved_from_archive`；训练/出集脚本拒绝读取归档的 v1 数据（`--allow-archived` 才放行，且打印横幅）
- [x] 8. 新版 `plan.md`、`claude_operation_log.md` 建立，顶部指向归档历史
- [x] 9. README 统一改写：删旧字段，只留当前能跑的内容 + 指向归档的一句话；`docs/design/CURRENT_PROGRAM_ARCHITECTURE.md` §7 数据布局同步
- [x] 10. 测试路径修正并全绿（core 348 / module 919 / full 1157 passed）→ 提交本阶段

**本阶段额外发现（见操作日志）**：`resolve_path` 对单文件条目的尾斜杠 bug；`batch_inventory`
在归档后再次出现"无导出文件"假零；`g7_label_transitions` 与若干测试因空读而静默退化为零/跳过。

**本阶段不做**：不搬任何代码文件（`experts/frequency.py` 等退役实现仍被包导出与测试导入，退役放到阶段 4 与导入方一起改）；不解除任何冻结门禁；不建新版运行时校准表。

## 1b. 阶段 1 收尾：根目录文档归类（2026-10-15，用户指定）

**规则**：根目录原则上只保留四份日常文档，其余按用途归类；**不改写历史资料、不改变运行逻辑**。

| 去处 | 文件 | 理由 |
|------|------|------|
| `docs/design/` | `active_forensic_agent_tasks.md`、`Reasoning_Framework.md` | 需求与设计资料留存，原始需求不改写 |
| `docs/design/` | `CURRENT_PROGRAM_ARCHITECTURE.md` | 继续描述实际架构；架构重大变化时同步更新 |
| `docs/design/` | `NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md` | 有效内容已整合进本文件（§1–§7 与 §2/§3/§4 的验收条件），原案留存避免重复维护 |
| `docs/archive/` | `ARCHIVE_INVENTORY.md`、`archive_inventory.json` | 归档预检快照（只作历史证据，不再重生成） |
| `docs/archive/` | `NEW_GENERATION_NAMING.md` | 代次/冻结/归档映射的规则记录，§6 已执行完毕 |
| `docs/development/` | `TESTING.md`、`docs/development/agent.md` | 测试与工程协作规范 |
| 根目录（保留） | `README.md`、`plan.md`、`claude_operation_log.md`、`KNOWN_ISSUES.md` | 四份日常文档 |

**`docs/development/agent.md` 的位置核实**：迁移前查过——没有任何工具依赖它的位置（`.claude/settings.local.json` 无 hook；
没有代码 `open()` 它；Claude Code 读的是 `CLAUDE.md`，根目录不存在该文件）。代码与文档里对
`docs/development/agent.md §x` 的引用都是**散文引用**，不构成路径依赖，因此**不保留根目录入口**，随本次一并移入
`docs/development/`，并把散文引用补上目录前缀。

**验收（执行结果，2026-10-15）**：

- [x] ①九个文件移动：`git status` 全部显示为 `R`（重命名），无内容增删
- [x] ②当前文档 20 条相对链接全部可解析（写入失效链接的自动检查：`tests/test_doc_layout.py`）
- [x] ③读文档路径的测试与脚本改到新路径：`tests/test_config_fingerprint.py`、
      `scripts/archive_inventory.py`（输出路径改为 `docs/archive/`）、`scripts/archive_v1.py`（读取清单的路径）
- [x] ④`core` 348 / `module` 927 / `full` 1169 通过（新增 8 项文档布局测试）
- [x] ⑤运行逻辑零改动（只改路径、链接与注释里的文件名）

**执行中发现并修掉的两处核验失效**（都是"核验本身不成立"而非数据问题）：
①`archive_v1.py --verify` 原来拿 `HEAD` 当移动前参照，移动提交之后 `HEAD:plan.md` 已是**新**计划，
于是把归档的 v1 计划误报为"内容变更"——改为以**新增归档路径的那个提交**（即移动提交）的父提交为参照；
②`archive_inventory.py --verify` 不认识文档归类，把移入 `docs/design/` 的提案误报为"缺失"——
改为按新位置说明并计数。两者现在都通过，且①在提交后仍然成立（比"必须提交前跑"更强）。

## 2. 已确认决策（实施中不再重开讨论）

来源：`docs/design/NEW_EXPERT_GROUP_REBUILD_PROPOSAL.md`（2026-10-15 方案，已确认部分）。
**本节是唯一在用的版本**；原案留存只作设计依据。

| # | 决策 | 说明 |
|---|------|------|
| D1 | 主模型迁移到 **Qwen3-VL-8B**（具体 checkpoint 待定） | 是视觉语言模型，**不是**纯文本 Qwen3-8B |
| D2 | 候选专家组 = **PROBE + noise（候选）+ NPR + UnivFD** | frequency_v2 与 jpeg **退出**；四专家是初始方案、不是硬约束 |
| D3 | **保留 v3 决策原则** | Qwen 主导调用、停止、结案；外部控制器只做执行、协议与边界安全控制，**不静默改判** |
| D4 | 新版运行时**关闭参考后验与预计收益** | 已决定，不再作为待选项：真正停止计算与提示注入，不是只在提示词里藏字段；离线对照独立保留 |
| D5 | 先采用**原始分数＋明确含义**基线 | 每专家一个主要原始分数＋尺度/方向/限制；不强制统一概率，不预先决定永不校准 |
| D6 | 证据包：**模型可见精简 / 后台记录完整** | 见 §3 的交付步骤 4 |
| D7 | 旧实验不进入新版 SFT | 新版数据只在 `sft_data/<generation>/` 生成 |
| D8 | 冻结与归档**不解除**门禁 | 训练入口拒收冻结来源与 `frozen_batch_used=true` 材料 |
| D9 | **不要同时大规模更换模型、提示词、专家和数据** | 一次只动一层，否则退化定位不了 |
| D10 | 分数语义必须**核实**后才能用 | 对照原论文/官方推理代码/实际权重核实字段的计算方式、尺度、正负方向、预处理与聚合，并用样例验证；**不按字段名推测** |

## 3. 实施步骤与验收条件

八步，顺序即依赖顺序（提案 §10）。每步的验收条件来自提案 §2/§4/§7/§8 的对应条款。

| # | 步骤 | 验收条件 | GPU |
|---|------|----------|-----|
| 1 | **归档与新代次目录** ✅ 已完成（§1/§1b） | 清单可追溯、摘要一致、新旧路径互不覆盖、新版批次命名不被旧登记误拦截 | 否 |
| 2 | **Qwen3-VL 迁移冒烟**（未开始） | ①单张图正确进入多模态输入且模型能描述具体场景；②输出一个合法工具请求且控制器真的执行了一次；③工具结果进入正确的后续消息且被准确引用；④调用之后仍有必要的图像上下文，结案可解析；⑤**模型原始回复、最终采用回复、外部控制原因三者分别留存** | 先 Mock（CPU），真模型需授权 |
| 3 | **候选专家独立评测**（未开始） | 固定权重/预处理/输出尺度与方向；同源图 × PNG/q70/q85/q95 分别评测；报告跨生成器表现、准确率、AUROC、**错误重合**与成本；判定是否提供**互补信息**；noise 同标准参评 | 是（需授权与权重） |
| 4 | **精简证据包 + v3 解耦**（未开始） | 运行时不再计算或使用参考后验与预计收益；不因旧后验/权重/校准表限制合法调用；重复请求不重复执行；边界截断原因可追溯；结论、报告与原始回复的差异有明确记录；方向检查改按**经核实的原始 score 语义**，不因校准字段消失而静默跳过 | 否 |
| 5 | **小批端到端验收**（未开始） | 视觉分析、证据忠实性、重复调用、安全停止、完整记录逐项过；**此阶段必须能人工查看原始图像、实际投递的证据包与完整原始对话**；通过后才允许扩大生成（可用最小审阅视图，不必等完整工作台） | 是（需授权） |
| 6 | **新版数据生成**（未开始） | 在新代次目录下生成；旧批次不混入；批次记录含模型/提示词/专家权重/预处理/schema/代码身份 | 是（需授权） |
| 7 | **工作台更新与人工审核**（未开始） | 核对 `KNOWN_ISSUES.md` 已修改项的**真实效果**，不只看自动通过率；原始对话 vs 实际监督内容对照、图片与分数语义、停止原因准确展示 | 否 |
| 8 | **SFT**（未开始） | 只有新版候选通过审核与训练门禁后启动 | 是（需授权） |

**测试与 GPU 纪律**：按 `core`/`module`/`full` 三层执行；GPU 任务逐次说明资源需求并获得授权，
**不以大量模拟测试替代真实模型验收**。

## 3b. 阶段 2 执行细则（2026-10-15，用户授权）

**已核实的事实（先核实再动手）**：

| 项 | 值 |
|----|-----|
| 模型 | `Qwen/Qwen3-VL-8B-Instruct`，**固定 revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`**（2025-10-15） |
| 许可 | **apache-2.0**（模型卡明示；与 PROBE 的"许可未声明"不同） |
| 体量 | 17.55 GB（4 个 safetensors 分片 + 处理器/模板文件，共 16 个文件） |
| 推理模式 | **non-thinking**：模型自带的两份模板（`chat_template.json`、`tokenizer_config.json`）都**没有** `enable_thinking` 开关——Instruct 版没有思考模式，要 thinking 得换 Thinking 变体。客户端在加载时复核这一点，不靠猜 |
| 依赖 | 本机 `transformers 5.14.0` **已含** `Qwen3VLForConditionalGeneration` / `Qwen3VLProcessor`；`torch 2.5.1+cu124`；无 `flash_attn`（用 sdpa/eager）；`huggingface_hub 1.23.0` |
| 下载方式 | `HF_ENDPOINT=https://hf-mirror.com` + **`HF_HUB_DISABLE_XET=1`**（镜像不代理 xet 的 CAS：走 xet 时权重分片 401，只有小文件能下——第一次失败即此）；**只进 HF 缓存，不设 `--local-dir`**，避免缓存与本地目录各存一份 |
| 落地文件系统 | `/`（overlay）：清理中断残留后 21.08 GB 可用，模型 17.55 GB，余约 3.5 GB；下载脚本带**看门狗**，可用空间低于 1.5 GB 即停止 |
| 用户后续动作 | 之后会删除 2.5-VL 旧模型（16 GB）给专家组腾位置——**未经确认不代删** |

**执行结果（2026-10-15 当日完成）**：

| 项 | 结果 |
|----|------|
| 权重下载 | ✅ 17.55 GB 全部落地（exit 0，0 个不完整分片），固定 revision，只进 HF 缓存 |
| CPU 协议测试 | ✅ 六项全过（Mock 客户端）；`core` 358 通过、`module` 938 通过 |
| **processor 预验证** | ✅ 真实 processor：2602 输入 token 中 **1024 个图像占位 token**，`pixel_values` [4096,1536]，两处 `image_token_id` 一致（151655） |
| **GPU 冒烟（真模型 + 真 PROBE）** | ✅ **六项全过**：①图像进输入（3817 token / 1024 图像 token）②一次合法调用（probe）③后续轮引用证据 `aigi_discriminator_score` ④结案可解析且收尾轮仍有图像 token ⑤三类记录分别留存 ⑥首轮视觉观察 125 字并另列疑似异常 |
| **实测显存** | 模型加载 **7.2 s**、权重显存 **17.53 GB**；整轮峰值 **18.6 GB**（4090 24 GB）；单会话 **53.4 s** |
| 判定正确性（单例） | Midjourney 图：PROBE P(Fake)=1.0，模型结案 Fake、confidence 0.984 ✓（真值 Fake） |

产物：`sft_data/g8_smoke/<时间戳>/{smoke_report.json,session_record.json}`；
`gpu_sessions_20261015/` 保存 GPU 会话的完整记录（含证据链）。

**实现清单（本阶段交付）**：

1. `mllm/qwen3_client.py`：`Qwen3VLClient`，复用既有协议实现（消息构建、格式纠错环、`ask()`），只替换模型类与处理器；
   **revision 固定进代码**；`inference_mode`（thinking / non-thinking）必须**验证模板真的支持**再设置，不支持就报错——
   不声称一个没设上的模式；权重身份（revision + 分片大小 + index 摘要）随客户端可取。
2. `scripts/g8_smoke.py`：最小冒烟入口，**只注册新版工具箱**（Mock 模式：`MockProbeExpert`；GPU 模式：真 PROBE + 一个固定 Mock 专家），
   产物写 `sft_data/g8_smoke/`，不进任何批次目录、不触发冻结门禁。
3. 五项断言（提案 §3）：①图像真的进入多模态消息（不是只给路径）；②一次合法调用被执行；③结果进入正确的后续消息且被准确引用；
   ④调用之后仍有图像上下文、结案可解析；⑤**模型原始回复、最终采用回复、外部控制原因三者分别留存**。
4. CPU 协议测试（`tests/test_g8_smoke.py`）：Mock 客户端跑通全流程并断言五项，纳入 `core`。
5. **GPU 验收（授权已获）**：真模型跑同一序列，验证图像输入、具体视觉描述、一次工具调用、结果引用、自主结案，并**实测显存**。

## 3c. 阶段 3：四专家独立评测（2026-10-15 起）

用户给定的口径（本轮确认）：

1. **重新处理全部源图，但保留数据划分**：约 9000 源图可由新模型与专家组重新处理，旧实验结果不混入；
   **不重新随机划分**——`train`（适配调试、生成 SFT）/ `val`（**专家选择、预处理与阈值选择、提示词验证**）/
   `test`（方案冻结后的最终验收，不进入 SFT 与开发决策）。**同一源图的 PNG/q70/q85/q95 变体必须同分区**；
   已参与历史开发的源图（旧 A/B/C 组与 `holdout`）如实记录，不重新冠名。
2. **参评：PROBE、noise、NPR、UnivFD**，Qwen 暂不参与融合。
3. **每次测量必须记录**：权重与代码身份、输入预处理、原始 score 及其**尺度与方向**、执行状态/耗时/错误。
   **Mock 只验证管线，不计入性能结果**。
4. **推进顺序**：CPU 接口准备 → 单图冒烟 → 小批专家评测 → 定专家与证据接口 → 小批 Agent 人工验收 →
   全量专家测分 → train 批量生成。

**已核实的事实（本轮，全部取自官方源码/实测，不是转述）**：

| 专家 | 事实 |
|------|------|
| **NPR** | 论文 *Rethinking the Up-Sampling Operations…*（CVPR 2024，arXiv 2312.10461）。分数 = `model(x).sigmoid()`，0–1，**高值偏 Fake**，官方阈值 0.5。预处理按配方开关：`no_resize`/`no_crop`、`Resize((loadSize,loadSize))` 或恒等、`CenterCrop(cropSize)`、**ImageNet 归一化**。模型 = 仓库 `networks/resnet.py` 的两段式 ResNet（只 layer1/layer2、`fc1=Linear(512→1)`、forward 内 `NPR = x - interpolate(x,0.5)`），参数 1.44M。**仓库根无 LICENSE（404）→ 许可未明确**。 |
| **NPR 权重**（两者架构相同、保存格式不同，**逐文件判定加载方式**） | A `NPR.pth`：17,393,733 B，sha256 `3939297e9399e0b992f87211610769d87d899de50d56da0204d6cbda2d483a53`，顶层含 `model/optimizer/total_steps` → **必须 `state_dict['model']`**（validate.py 写法；写成裸 dict 缺 146 键）。B `model_epoch_last_3090.pth`：5,842,385 B，sha256 `b67a91555ce786a6d0463ff0cb2b0b874d1c3f971b0e3febd2ae5618a80f7e8a`，裸 state_dict → **必须直接 load_state_dict**（test.py 写法；无 `model` 键）。**决定：A、B 都在 val 上测，由 val 选一个**（用户 2026-10-15）。C（GenImage 训练版）在 Google Drive，本机不可达，除非用户提供文件。 |
| **UnivFD** | 论文 *Towards Universal Fake Image Detectors…*（CVPR 2023，arXiv 2302.10174）。模型 = CLIP ViT-L/14 主干 + `fc_weights.pth` 线性头（实测 4083 B = 768×1）。预处理 **RGB → CenterCrop(224) → ToTensor → CLIP 归一化（无 Resize）**：mean `[0.48145466, 0.4578275, 0.40821073]`、std `[0.26862954, 0.26130258, 0.27577711]`。输出 = logit → `sigmoid` 0–1，**Real=0/Fake=1，高值偏 Fake**，默认 0.5 阈值，另有 `find_best_threshold`——**不得在最终测试集上搜阈值**。主干 URL 内嵌 sha256：`openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt`（933 MB，本机可达）。仓库许可 **MIT**（WAIV 2025）；主干归 OpenAI，条款另档。 |

**落地位置（用户建议，已建）**：`/root/autodl-tmp/third_party/{npr/{repo,weights},univfd/{repo,weights,clip_cache}}`。

**分数语义登记**（阶段三评测与阶段四证据检查**共用同一份**）：每个专家一条，含
`source_name / 权重身份(sha256+repo commit+file) / 预处理(名称+参数) / 原始分数定义(字段、范围、方向) /
阈值口径 / 许可状态 / 评测状态(Mock|已测) / 已知限制`。**sigmoid 分数不等于本地校准概率**，登记里必须分开写。

**本阶段交付**：
1. `scripts/g8_evaluate_experts.py`：统一离线入口（`--expert probe|noise|npr|univfd|mock-*`、`--split val|test`、
   `--partition`、`--limit`），逐图记录原始分与耗时/错误，产出可比对的报告；`test` 分区**拒绝阈值搜索**
   （阈值只能在 val 上选，冻结后才用于 test）。
2. 分数语义登记表 + 载入器（含"按文件判定加载方式"的机制：试两种写法，**都不行就报错**，不静默换一种）。
3. CPU 管线测试（Mock 专家与 Mock 适配器），GPU 小批评测在 val 上跑。

## 4. 视觉分析：生成前必须验收（提案 §8）

- 首轮包含**具体客观描述**，疑似异常另列，无异常可直说。
- **不强迫**模型编造"明显噪声""压缩块""传感器痕迹"。
- 工具调用后仍有必要的图像上下文；描述不在消息构造、转换或工作台里丢失。
- 保留模型原文，**不由转换器补写**模型未说过的场景。
- 先小批抽查原始输入与回复，通过后才扩大生成——不能等最终审核才发现视觉描述全没了。

## 5. 证据包与分数边界（提案 §5/§6）

**模型可见精简包**：专家名、证据 ID、执行状态、测量范围；一个主要原始分数及其尺度与方向；
简短结果说明与适用限制；可选证据图及其含义。
**后台完整记录**：工具、权重、预处理与 schema 版本；全部原始输出、中间量与错误信息；
执行时间、投递轮次、原始产物引用；图像生成方法、坐标、尺度与摘要。

- 分数、可选图片与文字作为**同一逻辑证据包**投递，不拆成互不关联的信息；
- **图片必须真正进入多模态消息**——只给文件路径不等于模型看到了图片；
- 不要求每个分类器生成热图；没有可靠空间含义时不制造解释图；
- 区分**未测量 / 已测但不确定 / 执行失败**；工具失败不得变成中性测量；
- 证据状态与引用结构在**生成前**确定，不能等工作台做完再补造原始信息。

必须在计划里写死、且容易被误读的几条（提案 §6.2）：

1. 原始 score **不自动等于** P(Fake)；不同专家的分数**不能直接比较或求平均**。
2. 区分 logit、sigmoid 分数、概率向量与聚合分数；不默认都是校准概率。
3. 不用含糊的"模型权重分数"，改称**独立验证性能或可靠性说明**。
4. **AUROC 是总体排序性能**，不是当前图像的判对概率，也不天然是融合权重。
5. 分类输出不得包装成物理伪影、相机来源证明或训练分布距离。
6. 池化校准本身不是缺陷；是否分格式由验证结果与样本量决定。
7. 实际输入的 JPEG 历史通常未知，不能假定总能拿到准确 q 值。

后续如比较校准方案：拟合集拟合、验证集选择、**独立最终测试**；评价概率质量与 Agent 的理解效果。

## 6. 仍需保留的通用问题（对照 `KNOWN_ISSUES.md`）

| 项目 | 新版需完成的内容 |
|------|------------------|
| 证据幻觉、字段与方向误读 | 查未执行专家、跨专家借用、泛称"其他专家支持"、Real/Fake 概率互换；复核误报与漏报 |
| 重复轮的训练处置 | 明确**整条拒收**或**逐轮排除/掩码**，不能只加徽章就当没在训练 |
| 工具故障与非法数值 | 加载失败、超时、NaN 及缺失结果如实记录，不伪装成正常返回 |
| 整条监督对话审计 | 中间轮与最终训练文本都审计；每轮只能引用此前投递过的证据 |
| 数据泄漏 | 同源及变体同分区；**已用于开发的旧 C 组不再冠名"未见最终测试集"** |
| 冻结训练门禁 | 新版训练拒收旧来源与冻结分析产物 |
| 配置版本绑定 | 模型、提示词、专家权重、预处理、证据 schema 与代码身份进入批次记录 |
| 审核工作台 | 新批次隔离；原始对话与实际监督内容对照；图片/分数语义与停止原因准确展示 |

旧专家专属的极性、阈值、校准与格式适用性**停止优化**，实际退出后按退役归档，不标为"已修复"；
旧批次补审不再是新版前置。

## 7. 硬性约束（任何阶段都适用）

1. **GPU 操作需事先明确授权**；核心逻辑必须能在 CPU 上跑通并测试。
2. **不推送远端**（本机到远端网络不可靠，只本地提交）。
3. **不复制、不重分发 PROBE / TruFor 的代码与权重**。
4. **不删除数据**（未经确认）；归档一律"移动 + 摘要 + 映射"，不覆盖、不改写。
5. 文件驱动流程：本文件先于代码、只追加；`claude_operation_log.md` 每条六字段；
   每个大段落更新 README；提交信息写清动机。
6. 根目录只保留四份日常文档（`README.md` / `plan.md` / `claude_operation_log.md` /
   `KNOWN_ISSUES.md`），其余在 `docs/{design,development,archive}/`（见 §1b）。

## 8. 待确认项（阻塞阶段 2–3，需用户提供）

1. ~~Qwen3-VL-8B 的具体版本与推理模式~~ **已解决**：`Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd0…`（apache-2.0），模板已核实无思考开关 ⇒ **non-thinking**；依赖与磁盘已核实（§3b）。仍待实测：显存占用。
2. **NPR / UnivFD 的权重、许可、预处理与输出定义**。
3. **noise 的评测口径与保留判据**（候选；退役与否由阶段 3 的增益与互补性结果决定）。
4. **重复请求轮的处置口径**：整条拒收，还是逐轮排除/掩码。
5. **新版独立最终测试源与小批验收规模**（旧 C 组已用于开发，不能再当未见测试集）。
6. 其余不清楚的问题可随时提出——本文件与 `KNOWN_ISSUES.md` 是讨论的落点。

_已完成、不再是待确认项：新代次目录与归档范围、历史路径恢复方式（`archive/PATH_MAP.json` +
`resolve_path`）、参考后验与预计收益的运行时关闭（已决定）。_
