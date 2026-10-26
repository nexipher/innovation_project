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
| 3 | **候选专家独立评测**（准备完成，待授权） | 固定权重/预处理/输出尺度与方向；首轮开发集 **100 源 × (original + jpeg_q95) = 200 输入/专家**，q85/q70 暂缓；报告跨生成器 AUROC+CI、官方阈值下召回/平衡准确率、失败数、耗时与内存；**错误重合只作描述性分析**，不设硬性准入线；noise 同标准参评 | 是（需授权；SPAI/AIDE 已就绪） |
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

**阶段三结论（2026-10-15，val 全量 284 源 × 4 格式 = 1136 张，错误 0）**：

| 专家 | val 整体 AUROC | 各格式 | 同设备成本 | 下一阶段角色（用户 2026-10-15 决定） |
|------|---------------|--------|-----------|--------------------------------------|
| PROBE | **0.9808** | 0.975–0.988（**不挑格式**） | 74.8 ms（GPU） | 主检测器，保留 |
| UnivFD | 0.8493 | 0.817–0.893 | 38.7 ms（GPU） | **进入小批 Agent 对照，尚不认定为必需专家** |
| noise | 0.6596 | 0.624–0.701 | 63.3 ms（CPU） | **退出新版默认工具箱**（历史实现与结果保留） |
| NPR-A | 0.6363 | PNG **0.9975**／JPEG 0.47–0.56 | 30.9 ms（GPU） | **PNG 条件下的补充候选**；JPEG 低分不得解释成强 Real 证据 |

**"是否提供互补信息"用源级配对 bootstrap 判定**（用户指正后的正确做法）：

| 组合 | 整体 Δ（源级） | PNG 格 | 结论 |
|------|---------------|--------|------|
| probe + npr-a | +0.0061 CI[+0.0025,+0.0108] P=0.000 | **+0.0228 CI[+0.0097,+0.0401] P=0.000** | **增益成立**，且集中在 PROBE 最弱的 PNG 格 |
| probe + univfd | +0.0030 CI[−0.0092,+0.0140] P=0.285 | +0.0093 CI[−0.0018,+0.0226] | **未证明增益，也未排除损害** |
| probe + noise | −0.0070 CI[−0.0215,+0.0059] P(Δ≤0)=0.838 | −0.0006 | 点估计为负但**未达显著**；退出理由落在弱判别力/格式敏感/语义负担，不是"已证明有害" |

**四条方法学纪律（本轮由用户指正写死）**：

1. **重采样单位是源，不是图**：1136 张变体来自 284 个源，按图独立重采样会把同一源的四份当独立样本，CI 约缩 √4。单格式格内每源恰一张，本无此问题。
2. **不允许"从不伤害"这类表述**：CI 含负值就只能说"未证明增益，也未排除损害"。
3. **固定等权 ≠ 没有选择偏差**：公式没有拟合参数，但在 val 上选 checkpoint／专家集合／使用条件是**方案选择**——这没问题，**前提是冻结后独立验收，且 test 不成为下一轮选型集**。
4. **组合层面的结论不得外推**：noise 的负增益只对"与 PROBE 等权平均"这一种组合成立。

**两条边界（同样写死）**：①**PNG 是容器，不代表没有 JPEG 压缩历史**——NPR 的适用条件写作"在当前 val 的 PNG 编码条件下表现强"，不是"所有 PNG 可靠"；②**计时必须同设备同范围**——NPR 的 271 ms 是 CPU、PROBE 的 74.8 ms 是 GPU，不能相减相比；同设备下 NPR **最便宜**（30.9 ms）。

**候选方案已冻结**：`sft_data/g8_eval/FROZEN_CANDIDATE_SCHEME.json`（角色、条件、成本、统计口径，以及 **test 的打开前置：证据接口、调用规则、提示词三者定稿之后**）。

**审计发现（阶段四输入）**：校准表按 `raw_metric` 建分箱（`utils/reliability.py::lookup`），而 token 同时带方向相反的 `strength`；`utils/evidence_ledger.py` 的数值核对**把两者任一都当作"模型引用的那个数"接受**——模型引用反向字段也能通过。阶段四的证据检查必须按语义登记判定模型指的是哪个字段。

**本阶段交付**：
1. `scripts/g8_evaluate_experts.py`：统一离线入口（`--expert probe|noise|npr|univfd|mock-*`、`--split val|test`、
   `--partition`、`--limit`），逐图记录原始分与耗时/错误，产出可比对的报告；`test` 分区**拒绝阈值搜索**
   （阈值只能在 val 上选，冻结后才用于 test）。
2. 分数语义登记表 + 载入器（含"按文件判定加载方式"的机制：试两种写法，**都不行就报错**，不静默换一种）。
3. CPU 管线测试（Mock 专家与 Mock 适配器），GPU 小批评测在 val 上跑。

## 3d. 阶段 4：接线与解耦（2026-10-15 起，用户指定五项）

顺序即用户给的顺序；每项独立提交、CPU 测试通过后再进下一项。**test 分区保持关闭**，
直到证据接口、调用规则、提示词三者定稿（见 §3c 的冻结方案）。

### 4.1 正式入口切换到 Qwen3-VL

`main.py` 目前 `--mllm qwen` 仍指向 Qwen2.5-VL。改法与本项目"当前即默认、历史须显式"的既有规矩一致：

- `--mllm qwen` → **Qwen3VLClient**（固定 revision，见 `mllm/qwen3_client.py`）；
- `--mllm qwen25` → 旧客户端，**仅用于 g7 复算**（与 `FORENSIC_HALTING_POLICY=v2` 同一性质）；
- 启动时把模型身份（repo/revision/dtype/推理模式）写进会话记录，便于批次版本绑定。
- 验收：`python main.py --image <img> --mllm qwen` 走 Qwen3；`--mllm qwen25` 仍可跑；
  `--dry-run`/CPU 路径不受影响（懒导入不被破坏）。

### 4.2 真正关闭参考后验与预计收益

现状：控制器仍加载校准表（`utils/reliability.py`），并计算 `posterior`、`tool_utilities`、
`conflict_score`、`redundancy_discounts`——冒烟报告里都能看到。要求"真正停止计算与提示注入"，
不是只在提示词里藏字段。

- **仅 v3 关闭**：计算与注入按策略版本分流；`FORENSIC_HALTING_POLICY=v1|v2` 的历史复算保持原样（独立保留）。
- 校准表在 v3 路径上**不再被加载**（`ReliabilityTable.load()` 不在 v3 初始化里调用）；调用旧字段的地方改为读登记表。
- 方向检查改按 **`experts/score_semantics.json` 的原始 score 语义**（字段、范围、方向、适用条件），
  不因校准字段消失而静默跳过；没有经验证的阈值/中性区间就不凭空设定。
- 验收（提案 §7）：v3 会话的报告里**不再出现** `posterior`/`tool_utilities`/`conflict_score`/`redundancy_discounts`；
  合法调用不因旧后验或校准表受限；重复请求仍不重复执行；边界截断原因可追溯；
  结论、报告与原始回复的差异有明确记录；v2 复算的输出与今天**逐字节一致**（回归）。

### 4.3 精简证据包与原始分数检查（PROBE / NPR-A / UnivFD）

- **模型可见**（按提案 §5 的表）：专家名、证据 ID、执行状态、测量范围、**一个主原始分数及其尺度与方向**、
  简短说明与适用限制、可选证据图及其含义。**后台完整记录**：权重/预处理/schema 版本、全部原始输出、耗时、投递轮次。
- 适用条件随分数一起给：NPR-A 写"在当前 val 的 PNG 编码条件下强，JPEG 格低分不等于强 Real"；
  UnivFD 写"中心裁剪 224，未看边缘"；PROBE 写"整图滑窗"。
- **原始分数检查**：账本按登记表判定模型引用的是**哪个字段**（修正 `raw_metric`/`strength` 任一都接受的缺陷），
  并核对数值与方向；引用反向或引用未投递字段 → 记为缺陷。
- 验收：三条证据的 token 字段集合固定且被测试锁定；乱写方向/数值/字段的构造样本全部被拒。

### 4.4 旧专家退出注册，更新提示词与调用规则

- 注册表只留 **probe / npr-a / univfd**；jpeg、frequency_v2 退出（**noise 也退出**，用户 2026-10-15 决定），
  实现与历史结果保留、可离线复算。
- 提示词按**登记表生成**（工具动作、测量范围、适用条件都来自同一份语义），不再是 v1 时代手写的三段表。
- 调用规则：保留单动作协议、请求去重、预算与超时；**判据不能用"证据不足"兜底**——工具缺席要如实呈现。
- 验收：默认工具箱身份明确；不加载未启用专家；完整 Agent 会话可运行；提示词里出现的每个专家都在注册表中。

### 4.5a 小批前的 CPU 核查（2026-10-15 完成，先于小批）

用户点名的三项，逐项做完并留下机械检查；**每一处都是先复现再修**：

| 用户要求 | 复查结果（修前） | 处置 |
| --- | --- | --- |
| 转换器适配精简包 | 账本只认完整 token 的字段名（`source`/`raw_metric`）。转换器喂给它的是 trace 里**当时投递的那段 JSON**，即精简包 → 整条被跳过：一次完全如实的引用被判 `fabricated_expert_result`（DEFECT）。**新主线的每个样本都会倒在审核这一关** | 账本认识两种形状（`ledger_view`：`expert`→`source`、`score.value`→登记表指定字段、方向取自逐值规则）；`_delivered_fields` 仍按投递戳算"模型能看到哪些字段" |
| 原始 `report` 不丢失 | 推理提示词要求 `<verdict>` JSON 带 `report`，而转换器渲染的答案里**没有这个字段**——训练出来的模型被教成写一个不含 report 的判决，与自己的说明相反 | 答案里带上**被采纳那一轮**的 `report` 原文（`adopted_report`，与 reasoning 同源）；那一轮没写就不含此字段，绝不代写 |
| 训练对话不重新注入后台字段 | 转换器用 `to_json(token)` 注入**完整 token**（`strength`/`raw_metric`/`calibrated_likelihood`/`reasoning`），即把模型没看过的字段塞回训练数据 | 按 token 自己的 `model_delivered` 重建投递形状；旧批次（无戳/`full`）保持不变以保证回放一致 |

复查过程中另外抓到的问题（都属"看起来跑通"这一类）：

1. **账本认不出新工具的名字**：`SOURCE_WORDS` 里没有 `npr`/`univfd`，模型写"NPR 的分数是 0.93"时整句话对账本**不可见**——不查数字、不查方向、不查编造。已补名字（与 probe/noise 同约定）。
2. **交付轮按次序对齐**：4.4 起会话可以在一次执行失败后继续，失败记录进证据链但**没有投递轮**，按次序对齐会把下一个 token 顶到上一轮的位置。改为按 `evidence_id` 认领（认不到时退回"第一个**可投递**的 token"）。
3. **状态小结把后台字段放回模型输入**：`_state_note` 原写 `P(Fake)=…／raw_metric=…／适用 …`，在精简包下面正好是被收起来的那些字段，而且与包**冲突**（包说 AI-generated，这行说"方向 —"，实测如此）。改为按投递形状说话（`_state_line`）。
4. **失败包带着"带方向的空分数槽"**：`model_visible` 对失败 token 仍给出 `score`（`value: None` 却带 `range`/`direction`/`threshold`），读起来就是一次中性测量。失败包现在**没有分数、没有尺度与方向**。
5. **登记表里混着两类读者**：`prompt_note` 与内部说明（`usage_conditions`/`limits`/阶段编号）曾经同格；noise 的"退役…仅供回放"与 `G7 fitting group` 这类台账词汇会随提示词送到模型面前。已按字段分家，并加**机械扫描**（`test_score_semantics.py` 的两类读者 + `scripts/g8_dump_model_inputs.py` 的词表扫描）。
6. **空段头描述着本次看不到的工具**：当前工具组全部由登记表供词，手写的 `WHAT EACH TOOL ACTUALLY MEASURES` 段正文为空，段头却还在（且带阶段编号）。空段现在整段不出现。

**证据（用户要的"打印几份实际模型输入"）**：`python scripts/g8_dump_model_inputs.py` →
`sft_data/g8_model_inputs/<时间戳>/model_inputs.md`，内容为系统提示词原文、一次真实会话的全部对话（CPU/mock）、逐值证据包（含 NPR 低分＝`Uncertain` 的边界与失败包）、内部标记词表扫描。**当前扫描结果：0 命中。**

### 4.5b 小批的生成入口（`scripts/g8_generate_batch.py`）

现成的 `generate_trajectories_g4.py` 是 **G4 多策略矩阵**的入口：它的策略表是旧工具箱（noise/jpeg/freq/probe）、客户端是 Qwen2.5、还要过 applicability 门禁——拿它跑 g8 小批意味着"验收看的管线"和"生成数据的管线"不是同一条。按 §5"必须在计划里写死"的精神，小批用**自己的入口**：

- **与验收入口同源**：会话由 `main.build_pipeline("qwen")` 构造（CUDA 门禁、专家不可用即拒、设备自动解析并打印），不是另写一份等价物；
- **闸门照旧**：`assert_not_frozen_batch`（生成新轨迹）、真实会话需 `FORENSIC_ALLOW_GPU=1`、`--dry-run` 走 mock（CPU 管道验证），`--limit` 可小跑；
- **可续跑**：已存在记录的变体跳过（断电/中断不必重跑）；
- **选源不藏**：变体来自 `build_variants_g4.py`（`--partition train`、类平衡抽样、固定 seed），清单与计数写进批次 `metadata.json`，与记录里的 `split/treatment/source_id` 对得上；
- **批次身份写进 `metadata.json`**：模型 revision 与精度、提示词 digest、三个专家各自的**设备**与权重身份、停机策略、投递形状（`model_delivered`）、以及"这批数据是用哪条入口跑的"。批次名 `b1`（用户 2026-10-15 选定），规模 **8 源 × 4 格式 = 32 会话**。

### 4.6 g8 审核工作台（验收专用，不改 b1 事实）

用户 2026-10-15 的八条要求。**只改展示与审核功能**：b1 的记录、trace、判定一律不动、不重跑模型。

**① 审核身份与失效**：决定绑 `(generation, batch, 样本内容指纹)`。指纹＝对**页面展示的内容**（记录里的判定/停止原因/计数器/引用检查 + trace 里逐轮原文）做规范化 JSON 的 sha256 前 16 位。重渲染而内容不变 → 指纹不变 → 决定保留；内容变了 → 指纹变 → 旧决定标**已过期**，**不自动继承**，也绝不悄悄沿用。导入时 generation/batch/指纹三者任一不匹配即**拒绝**（跨代次、跨批次导入一律拒）。
**② 存储两层**：浏览器 `localStorage` 只是**草稿**；写回仓库要走 `scripts/g8_review_import.py <导出文件>`，它经同一套校验后追加到 `sft_data/g8/review/<batch>/dispositions.jsonl`。页面必须写明"浏览器里保存 ≠ 已写回仓库"。**不复用** `sft_data/review/dispositions.jsonl`（那是训练口味的处置台账）——验收与训练分开。
**③ 验收 ≠ 训练**：按钮叫「通过（本次审核）」，只表示这一条**行为**通过；页面说明它**不会**让样本变成可训练，也不接 `admit_trajectories_g4`（那会把 32 条全挡在外面）。
**④ 原文可核对**：区块名改为「已记录的会话原文」；助手文本按 trace 原样展示、**不总结不补写**；逐轮标注来源（模型输出／工具返回（投递包）／控制器提示／任务下发）；JSON 除美化外给**原始文本** `<details>` 与 **trace 下载链接**。
**⑤ 投递与后台分开**：主区展示 **trace 里保存的实际投递包原文**（不再用当前 `model_visible()` 重新生成冒充历史输入——那是拿今天的代码改写过去的输入）；后台完整证据（token 全字段）另设折叠区。摘要可截断，但必须有完整展开入口。
**⑥ 审计信息补齐**：原图与**实际输入变体**（两张）、真实尺寸（并写明来源）、模型与专家版本（批次 `metadata.json`）、停止原因、请求/执行/重复计数、**引用检查的具体发现文本**（不只数量）；未记录的项标「未记录」。
**⑦ 日志完整性**：页面给每批一张"日志完整性"表，逐项标已记录/未记录。**b1 缺三项**（初始 system prompt、每次调用的多模态输入、客户端内部重试）——如实标注，**不事后重建冒充原始日志**。后续批次补记：客户端累积每次调用的输入统计与每次尝试的原始输出（含被格式校验拒掉的那次），控制器在结束时把这些写进 trace metadata。
**⑧ 验证后再交付**：测试覆盖进度保存（草稿合并语义）、导入导出隔离、内容变更失效、原文一致性（页面里助手文本与 trace 逐字相同）、图片链接可达、以及"审核不触发训练"（页面与导入路径都不写训练产物）。

### 4.8 b2 预注册：协议修订的配对验证（跑之前写死）

b1 的人工审核（14 通过 / 18 待修）指向的是**措辞与引用纪律**，4.7 据此改了协议。b2 的唯一目的就是验证这些修订是否改变模型行为，**不是**扩大生成。

**设计**：`g8 / b2`，与 b1 **同一批 8 源 × 4 格式**（同一 seed，配对），32 会话；**输入文件是 b1 的逐字节副本**（`sft_data/g8/variants/b2/` 由 b1 复制而来，每条记录 sha256/bytes，核对 32/32 一致）——不靠"同 seed 重建"来假定相同；除提示词外全部条件相同——Qwen3-VL-8B@`0c351dd0`、v3、精简包、probe/npr/univfd、贪心、非 thinking、固定 revision。提示词指纹：b1 = `af7003e17e0d6a42`，b2 = `8d664a768bb74e9d`（`FORENSIC_SYSTEM_PROMPT` 哈希 `a01272f96bbd66c7 → 259069a1ddb66348`）。

**判据（先写死，跑完用 `scripts/g8_protocol_audit.py` 同一口径数，不许事后改）**：

| 判据（只看模型输出轮） | b1 实测基线 | b2 目标 |
| --- | --- | --- |
| 疑罪从无 / presumption of innocence | 4 处 / 4 会话（12%） | **0 处**（提示词的邀请已删；再出现是模型自带） |
| "无其他专家冲突／证据链足够强／已交叉验证" | 19 处 / 18 会话（56%） | **≤ 6 个会话** |
| "压缩/编码/噪声已被排除或无影响" | 23 处 / 16 会话（50%） | **≤ 6 个会话** |
| 结案声明证据范围（专家数/是否跨专家对照） | **0/32（0%）** | **32/32** |
| 判定与真值一致 | 32/32 | ≥ 32/32（不得下降） |
| 引用检查缺陷 / 投递形状 | 0 缺陷；全 slim | 0；全 slim |

**防"机械加一句"**：判据只数"有没有说"；另人工抽查 8 条（每生成器/格式各一条）核对"说得对不对"，报告里引用原句。b2 仍属**行为验收**，不自动可训练。

### 4.8b b2 结果与判据口径更正（2026-10-18，跑后记录，不改预注册目标）

**b2 实跑（32 会话，输入与 b1 逐文件相同，仅提示词不同）**：

| 项 | b1 | b2 |
| --- | --- | --- |
| 正确 / 错误 / 弃权 / 未形成结论 | 32 / 0 / 0 / 0 | **27 / 0 / 5 / 0** |
| 工具调用分布 / 停止原因 | 1×32 / model_concluded×32 | 1×32 / model_concluded×32 |
| 耗时（均 / 最长） | 27.5 s / 43.2 s | 26.2 s / 36.5 s |
| 日志完整性（系统提示词 / 调用明细 / 尺寸） | 0/32 / 0/32 / 32/32 | **32/32 / 32/32 / 32/32** |
| presumption（会话） | 4（12%） | **0** ✓ |
| overclaim（会话） | 19（59%） | **0** ✓ |
| excluded（会话） | 2（6%） | **0** ✓ |
| 结论轮声明证据范围 | 0/32 | **32/32** ✓ |

**未达标的预注册项**：判定一致率 27/32 < 32/32。5 条弃权**全部是 Fake 且 probe ≥0.98**（0.980–0.9998），理由均为"仅一个专家提供测量、未做跨专家对照"。按用户口径弃权单列、**不计为错误**；但"是否接受为合理弃权"属于人工验收判断——若不接受，协议需要下一轮修订（例如明确"单一有效测量在结案时可以足够，只要说明未做对照"）。

**判据口径更正（如实记录）**：初版检测器把两类**正确行为**误算成问题——"无其他专家提供交叉验证"（新协议要求的诚实声明）与"未发现压缩伪影"（RULES 8 允许的视觉观察）。更正后：b1 的 excluded 从 23 处/16 会话降为 **2 处/2 会话**（与人工审核点过的 3 条基本吻合），overclaim 19→**19** 会话（语句形态不变），scope 仍为 **0**。检测器现在按**意图**判定（物证词不算点名专家；诚实缺席不算吹嘘；观察不算排除断言），并加了正反例测试。**预注册不达标项的结论不受口径更正影响**（b2 的 5 条弃权是结果层面的，不是检测器造成的）。

### 4.9 外部泛化评测（2026-10-18 用户定稿）

与 b2 **两条独立任务线**：b2 验证协议修订（b1 的 32 个输入），这一条测外部图上的泛化。**两条线都不自动产生可训练 SFT 数据**（批次 `purpose` 字段分别记 `protocol_acceptance_only` / `external_evaluation_only`）。

**输入：100 张固定源图 × 2 个条件 = 200 个输入文件**（`dataset/mnw_div2k/inputs_manifest.json`）：

| 条件 | 处理 |
| --- | --- |
| `original` | 下载文件原样引用，不改动（哈希与固定清单逐一核对） |
| `jpeg_q95` | 真假两半用**同一编码器、同一参数**重编码，不缩放 |

编码器身份写在清单里（Pillow 11.0.0｜libjpeg 6.2｜quality 95｜subsampling 2｜optimize/progressive 关）。统一 JPEG 只控制**最终编码流程**：MNW 源本身是 JPEG（历史压缩代数未知）、DIV2K 是 PNG，两个来源的编码代数不同；**不得**把 JPEG→PNG 解释为未压缩，也不把该条件当作"消除格式混杂"。

**对象与次数（完全相同的输入）**：PROBE 200、NPR-A 200、UnivFD 200（共 600 次专家测量）；完整 Agent **200 会话**（不是 50 张子集）。执行顺序：**专家基线先跑**；Agent 在 b2 验收、协议冻结之后再跑。专家分数**不预注入** Agent 输入——Agent 只看到自己实际调用后返回的证据，内部工具调用另计。

**分析（按生成器 × 条件）**：每组 = 某生成器 20 Fake + 同一组 20 DIV2K Real（n=40，单标签组算不了 AUROC）；20 张 Real 在四个生成器组间复用 → 组间不独立，只作组内解释。两种条件作用在同一批源上 → **按源配对**比较（配对 ΔAUROC，bootstrap 按源重采样），不把两种条件当独立样本比 CI。指标：组内 AUROC+95%CI、按登记表既有阈值判对的数、Fake/Real 分数分布；格式与分辨率分别列出；不跨组汇总、不给单一总分。

**用途限制**：仅外部评测；不用于 SFT、阈值拟合或专家选型后反复调参；不宣称逐图绝对未见（DIV2K 为学术用途、MNW 仅非商业评测）。真实/合成在来源、分辨率、题材上三重混杂，报告必须按组与条件分开。

**成本**：Agent 200 会话按 b1 实测约 27 s/会话 ≈ **1.5–2 小时**（不含审阅）；专家 600 次测量在分钟级。

**入口**：专家基线 `scripts/g8_external_eval.py`（读 `inputs_manifest.json`，逐条写 `scores.jsonl`，可 `--resume`；报告按 生成器×条件 给出 AUROC+95%CI、登记表阈值判对率、以及两种条件的**配对 ΔAUROC**）。它不训练、不调阈值，报告头写明用途限制。

### 4.9b 外部专家基线结果（2026-10-18 跑后记录）

**运行**：`python scripts/g8_external_eval.py --device cuda`，600 行分数写入
`sft_data/g8_external_eval/20261018_202449/scores.jsonl`（sha256
`a4937e110e7c139f05779ec362f8115e7af3d621f4178e55a0f2dc9a166a58bf`，600 行，未重测）。
单次耗时中位数：PROBE 209.8 ms、NPR 57.4 ms、UnivFD 55.4 ms。

**报告口径修正（跑后发现，只改报告不改分数）**：第一版报告按"来源组 × 条件"分组，
而每个来源组只有单一标签（生成器组 20 Fake、DIV2K 组 20 Real），AUROC 全为 `None`。
按 §4.9 的设计改为**跨组配对**：每组 = 该生成器 20 Fake + 同一条件 20 DIV2K Real
（n=40）；配对键是**源**（`source_id`/`source_path`），不是带条件后缀的
`variant_id`（`__original`/`__jpeg_q95` 跨条件永不相等——第一版重建曾因此得到 0 对）。
报告可由 `--report-only <scores.jsonl>` 从既有分数重建，不重跑测量。

**输入构成（报告 `composition` 字段）**：`original` 条件 Fake=80 JPEG / Real=20 PNG；
`jpeg_q95` 条件 Fake=80 JPEG / Real=20 JPEG。**标签与容器在 original 条件下完全混杂**，
这是 DIV2K 原图为 PNG、MNW 原图为 JPEG 造成的，报告必须带着这个前提读。

**组内 AUROC（n=40，95% bootstrap CI）与登记表阈值判对率**：

| 组 × 条件 | PROBE | NPR-A | UnivFD |
| --- | --- | --- | --- |
| Adobe_Firefly_v4 / original | 1.000 [1.000,1.000]｜acc 0.975 | 0.203 [0.060,0.368]｜acc 0.325 | 0.815 [0.664,0.942]｜acc 0.75 |
| Adobe_Firefly_v4 / jpeg_q95 | 0.995 [0.977,1.000]｜acc 0.95 | 0.198 [0.057,0.348]｜acc 0.475 | 0.838 [0.693,0.945]｜acc 0.75 |
| ByteDance_Seedream_v4 / original | 1.000 [1.000,1.000]｜acc 1.0 | 0.135 [0.025,0.275]｜acc 0.325 | 0.493 [0.299,0.679]｜acc 0.475 |
| ByteDance_Seedream_v4 / jpeg_q95 | 1.000 [1.000,1.000]｜acc 0.975 | 0.100 [0.021,0.202]｜acc 0.475 | 0.555 [0.361,0.737]｜acc 0.475 |
| Google_Imagen4 / original | 0.995 [0.975,1.000]｜acc 0.95 | 0.140 [0.033,0.267]｜acc 0.325 | 0.373 [0.188,0.556]｜acc 0.475 |
| Google_Imagen4 / jpeg_q95 | 0.990 [0.960,1.000]｜acc 0.925 | 0.135 [0.030,0.263]｜acc 0.475 | 0.448 [0.248,0.635]｜acc 0.50 |
| Midjourney_v7 / original | 0.995 [0.973,1.000]｜acc 0.925 | 0.151 [0.037,0.284]｜acc 0.325 | 0.393 [0.212,0.578]｜acc 0.475 |
| Midjourney_v7 / jpeg_q95 | 0.990 [0.960,1.000]｜acc 0.90 | 0.250 [0.099,0.422]｜acc 0.475 | 0.400 [0.228,0.591]｜acc 0.475 |

**配对 ΔAUROC = jpeg_q95 − original（按源配对，bootstrap 按源重采样）**：

| 组 | PROBE | NPR-A | UnivFD |
| --- | --- | --- | --- |
| Adobe_Firefly_v4 | −0.005 [−0.023, 0.000] | −0.005 [−0.194, 0.176] | +0.023 [−0.023, 0.071] |
| ByteDance_Seedream_v4 | 0.000 [0.000, 0.000] | −0.035 [−0.194, 0.113] | +0.063 [0.000, 0.135] |
| Google_Imagen4 | −0.005 [−0.023, 0.000] | −0.005 [−0.149, 0.126] | +0.075 [0.008, 0.162] |
| Midjourney_v7 | −0.005 [−0.028, 0.005] | +0.099 [−0.061, 0.273] | +0.008 [−0.051, 0.073] |

**读法（逐条限制写在这里，不允许下游丢掉）**：

1. **PROBE 在全部 8 个组×条件上 AUROC 0.99–1.0**，阈值 acc 0.90–1.0，jpeg 重编码的
   配对 Δ 均不显著（CI 含 0）。但**这是第三方 AIGI 检测器**（ModelScope
   `shuinishaojiu/PROBE-AIGI-Detection`，上游未声明训练语料）："外部"只表示**未参与
   本项目的专家选择/阈值选择**，不表示其上游训练集未见过 MNW 类图像；不得写成
   "未见生成器泛化"，也不得据此宣称本项目的方法有效。
2. **NPR-A 的分数在本外部集上方向反转**：AUROC 0.10–0.25，全部 80 张 MNW 假图
   **精确为 0.0**；DIV2K 原图（PNG）均值 0.3634、35% 过阈值；同一批 DIV2K 转
   jpeg_q95 后均值 0.0451、5% 过阈值。这与 b1/val 上"PNG 格 AUROC 0.9975、JPEG 格
   0.47–0.56"一致：**NPR 的分数跟着容器/压缩史走，而不是跟着生成器痕迹走**。
   b1 val 的 PNG 高 AUROC 是格式混杂下的表观值，不能当作外部检测能力。NPR 仍可
   在**明确标注适用格式**的域内证据里使用，但其证据必须带 `measurement_scope`
   与格式条件，不能外推。
3. **UnivFD 在四个生成器上 3 个接近或低于随机**：Adobe 0.815–0.838，ByteDance
   0.493–0.555，Google 0.373–0.448，Midjourney 0.393–0.400；假图均值大多 ≤0.05，
   DIV2K 真图均值 0.113（高于假图）。其 b1 独立判别力 0.849（0.817–0.893）没有
   迁移到这批外部生成器。jpeg 条件对 Google 的 Δ 为 +0.075（CI 0.008–0.162），
   但两种条件的 AUROC 都**低于随机**——不构成"条件使专家可用"的证据。
4. 配对条件只统一**最终编码流程**，不消除历史压缩代数、来源、题材、分辨率差异；
   不把 jpeg_q95 当作"消除格式混杂"的条件。20 张 DIV2K Real 在四组间复用，
   组间不独立，只作组内解释；不跨组汇总、不给单一总分、不拟合阈值。
5. 这批结果**不产生可训练 SFT 数据**，也不据此调专家阈值或挑选专家。

### 4.12 执行顺序与**硬暂停点**（用户 2026-10-18 定稿）

顺序是死的，且第 ③ 步不是"建议"：

1. **检查 + 真实冒烟**：GPU 可用、三专家权重可加载、b2 的 32 个输入与 b1 逐文件相同 → 冒烟六项。
2. **b2 32 会话** → 重建审阅页，跑 `g8_protocol_audit.py --batch b1 --batch b2` 出并排对照。
3. **暂停，等人工验收**：全部命中/错误/弃权逐条复核 + 抽查 8 条未命中。**不自动继续。**
4. **外部专家基线可独立先跑**（`scripts/g8_external_eval.py`，600 次测量）——它不依赖 Agent，也不改协议。
5. **b2 验收通过、协议冻结之后**，才跑外部 Agent 的 200 会话。

**冻结是用户的显式动作**（写进 `sft_data/FROZEN.json`），代码不自动冻结。第 5 步的前置条件做成了**门禁**而不是文字：`g8_generate_batch --purpose external_evaluation_only` 在启动前调用 `assert_protocol_accepted()`，要求 (a) `g8/b2` 已在 FROZEN.json 里，(b) b2 验收台账每条的最新决定都是 `pass` 且不过期；缺任一项直接拒绝（`SystemExit`），不会加载模型。冒烟、b2、专家基线都不受这道门影响。

### 4.12b 外部 Agent 门禁：逐样本覆盖、现场指纹与协议身份（2026-10-18 用户指定）

现有门禁 `assert_protocol_accepted()` 只看两件事：`g8/b2` 在 `FROZEN.json` 里，以及
台账决定数不少于记录数。这留下四个洞：

1. **按数量比对**：台账里其他样本的决定可以凑数；重复 `trajectory_id`、空批次、
   损坏记录都不会被明确拒绝。
2. **相信台账的 `stale=false`**：导入时标过一次就不再重算，内容后来变了也放行。
3. **把"批次冻结"当成"协议冻结"**：b2 冻结只说明那 32 条数据是历史；外部 Agent
   实际使用的主模型 revision、提示词、专家权重/预处理、证据包、预算与调用规则
   可能已经变了，却没有东西把它们和验收时绑定。
4. **产物的出现顺序**：检查若发生在目录创建/模型加载之后，失败会留下看起来像
   正常结果的批次目录。

**门禁设计（四条全部满足才放行）**：

1. **用户明确冻结**：`FROZEN.json` 必须可读、可解析、`frozen` 是列表，且含
   `(generation, batch)`。文件缺失/损坏 = 明确拒绝，不再当成"未冻结"。
2. **逐样本覆盖**（`utils/acceptance.py`）：以记录目录里的真实 `trajectory_id`
   为准，逐个与台账最新决定匹配。
   - 记录目录不存在、为空、文件无法解析、缺 `trajectory_id`、跨文件重复 ID → 拒绝
     并列出文件；
   - **期望样本集**由 b2 的变体清单（32 个 `variant_id`）+ 批次 metadata 的
     `policy` 还原为 `{policy}__{variant_id}`，与实际记录逐 ID 比对：记录被删掉
     一条不能变成"当前样本变少了"，多出一条同样拒绝；
   - 每条记录必须能读到 `trace_path`、trace 文件存在且可解析；否则拒绝并列出 ID；
   - 台账不存在、为空、某行无法解析、某行的 `generation/batch` 与当前不符 → 拒绝；
   - 当前样本缺审 → 列 ID；最新决定是 `revise` / `reject` → 分别列 ID；决定字段
     不合法（无审核人、无指纹、未知决定）→ 拒绝；
   - 台账里属于当前批次不存在的 ID **不计入覆盖**，单独列出但不单独构成拒绝
     （台账是追加式的，历史 ID 可能还在文件里）；当前每条样本仍必须有自己的有效
     通过决定。
3. **现场指纹**（复用工作台 `sample_fingerprint(record, trace)`）：对每条当前记录
   与 trace 重新计算指纹，与最新决定的指纹比较。不相等 → 拒绝并给出两个指纹。
   **不看台账的 `stale` 字段**：那是导入时的旧判断；当前指纹才是事实。不自动改
   决定、不自动改成通过、不重写台账。
4. **协议身份**（`utils/protocol_identity.py`，schema
   `acceptance_protocol_identity/1`）：
   - 内容：主模型（checkpoint、revision、dtype、inference_mode、解码参数）、
     系统提示词（digest + 实际 sha256）、专家集合（每个专家的 class、权重
     sha256/字节数、登记表预处理）、模型可见证据包（版本 = 对
     `EvidenceTokenizer.MODEL_VISIBLE_KEYS` 的哈希）、工具与预算（policy、
     tools_served、HALTING_POLICY、MAX_EXPERT_CALLS、MAX_MODEL_TURNS、
     evidence_injection）、协议文件内容哈希（controller / evidence_tokenizer /
     message_builder / qwen3_client / config / score_semantics / external_base +
     每个专家适配器）；
   - **登记**（`scripts/g8_accept_protocol.py`）：只在逐样本覆盖与指纹全部通过后
     运行；从 b2 自己的 `metadata.json` 取身份，并核对每个协议文件的当前字节与
     **b2 运行提交**中的字节一致——不一致就拒绝登记（b2 验收的不再是当前协议）。
     写 `sft_data/<gen>/review/<batch>/accepted_protocol.json`（最新）并追加
     `accepted_protocol_history.jsonl`（历史版本）；不改 `FROZEN.json`，冻结仍是
     用户的显式动作；
   - **运行前比对**：外部 Agent 构建完真实管线（`main.build_pipeline`）后，用
     **实际对象**生成身份（不读 CLI 声明、不读 b2 元数据），逐字段与验收身份
     比较；任何字段不一致 → 拒绝并指出字段的验收值/当前值。没有验收身份文件
     同样拒绝。
5. **顺序与产物**：第一阶段（冻结 + 逐样本 + 指纹 + 验收身份存在）在创建任何
   批次目录之前执行；第二阶段（实际协议身份比对）在 `main.build_pipeline` 之后、
   `os.makedirs(out_dir)` 之前执行。任一步失败 → 不创建记录/trace 目录、不写
   `metadata.json`，不会留下看似正常的批次产物。b2 本身、冒烟、独立专家基线不
   经过这道门。

**验收条件**：逐样本覆盖/重复/空批/坏台账/坏 trace/指纹不符各有反例测试；协议
身份在模型 revision、提示词、专家权重、证据包、预算、解码、协议文件哈希不一致
时都必须拒绝且指出字段；门禁失败后目标批次目录不存在；全部通过时 `run_batch`
才继续。

### 4.16 b2 人工审核问题确认（2026-10-18，只记录，不修复）

**输入**：`sft_data/g8/review/b2/review_g8_b2.json`（reviewer `hj`，exported_at
`2026-10-18T13:01:11`，32 条决定，27 revise / 5 pass）。32/32 条决定的内容指纹与当前
b2 记录 + trace **逐条一致**；该文件**尚未导入台账**，b2 仍未通过验收、未冻结。
本节只确认问题，不改写 b2 记录、不自动标通过、不改审核决定。

**总体读法**：5 条 pass（#10/#11/#15/#19/#20 的顺序号见导出排序）说明"单专家、
未交叉验证"本身**不是**问题——这 5 条明确限定了证据范围、没有编造局部结果、结论
与当前证据一致。问题集中在：**没有依据的停止/可靠性声明、把全局测量改写成局部
结论、替未调用的专家断言、以及未经测量的机制推断**。

**A. 模型行为（27 条 revise 均至少命中一类；下述可重叠）**

| 子类 | 条数 | 会话（variant 简称） |
| --- | --- | --- |
| A1 单专家提前结案 / 以"无冲突、无其他专家"作为停止或可靠性依据 | 16 | BigGAN q70/png、Glide q70/q85/q95、Real_0bb q70、Real_532 q70/q85/png、Real_61a q70、Real_f4141 q70/q95/png、VQDM q70/q85、Wukong q85 |
| A2 全局测量被改写成局部结论（"未见任何区域异常/局部异常""未受局部区域影响"） | 11 | BigGAN q70、Glide q70/q95、Real_532 q70/q85/png、Real_61a q85、VQDM q70/png、Wukong q70/q85 |
| A2b 替未调用的专家断言"不存在可被专家检测的伪影" | 3 | Real_f4141 q70/q85/png |
| A3 未经测量的机制推断（PROBE 对低分辨率敏感/不受边缘影响、传感器、与训练图像高度匹配、误判来源猜测、从"可能"升级为"存在"） | 7 | BigGAN q85/q95、Glide q85、Glide png、Wukong q70/q85/q95 |
| A4 校准/概率语言混用与过强措辞（"高概率 AI 生成"与"未校准"并置；"决定性证据/结论可靠/高置信度"） | 5 | Glide png、Real_f4141 q70、VQDM q95/png、Wukong q70 |
| A5 视觉场景补写 | 1 | Real_0bb png（"天空晴朗"；原图上半部是蓝色远山/林坡与浅色斑块，未见天空——已放大核对） |
| A6 自我矛盾（"需进一步验证"后立即结案） | 3 | BigGAN png、VQDM q70/q85 |

**特别核实的两条**：

* **#26（VQDM q85）**：trace 的 `available_experts` 明确列出 `npr_expert`、
  `univfd_expert`，模型却写"无其他专家可提供交叉验证"后结案；这是**事实性错误**，
  不是措辞问题。
* **#32（Wukong png）**：b2 记录的 `evidence_check` 把"该专家的高分表明图像整体结构
  或语义存在AI生成特征"归因为 `jpeg_expert`（本会话未执行），属**控制器误报**。
  用当前代码重审同一 trace（纠正前后两轮）均为 **0 findings**——该误报在 b2 之后
  的代码里已修复，b2 记录按原样保留。模型在纠正后重复了同一段 reasoning（未改写），
  但最终 report 不含该句；纠正循环没有强制"改写到不再命中"。

**B. 证据包**

1. 字段本身没有发现引用错误：5 条 pass 与多数 revise 的分数、方向、阈值引用正确；
   失败包/阈值/方向/`is_calibrated_probability` 语义没有暴露新的字段级缺陷。
2. **`scope` 与"locates nothing"已在包内**（每条 probe 投递包都有），模型仍在
   11+3 条里改写为局部/替专家断言——这不是"缺一句话"，而是**包内长文本约束没有被
   结构化执行**。下一步需要的是结案时的 scope 声明/校验，而不是再加一段说明。
3. `reading` 中的 "Strongest available evidence" 可能被模型读成"决定性/最强证据"
   （#29/#32 的措辞与之呼应）；这是**证据包措辞**问题，列入下一轮修订候选。
4. 另一条包内语义（"one measurement is not a cross-expert comparison"）没有被
   遵守：模型把"未调用"写成"无冲突/无其他专家"。这属于模型行为，但提示词/包的
   位置与强度需要一起评估。

**C. 控制器**

1. **C1 证据账本误报（#32）**：已修复（当前代码 0 findings），但 b2 记录保留旧的
   `failed`；工作台仍展示旧结果（见 D2）。
2. **C2 纠正循环**：模型在收到纠正后重复同一 reasoning，控制器记录
   `after_correction` 仍命中并结束会话；没有"纠正后必须变化/否则升级"的机制。
3. **C3 可用性一致性**：状态提示列出"尚可调用：npr_expert, univfd_expert"，模型
   仍可写"无其他专家"；控制器没有检查"结论声称无专家 vs 可用专家非空"。
4. 预算/停止策略（v3：模型主导、最多 5 次调用/6 轮）允许 2 轮 1 次调用后结案；
   这不是实现 bug，而是**策略选择**。27 条 revise 中 16 条指向该策略的后果，
   下一轮协议修订需要重新定义"何时必须继续取证"。

**D. 工作台 / 自动审计**

1. **D1 自动协议审计覆盖不足**：`g8_protocol_audit.py` 只把部分"无交叉验证"句
   标为 scope 候选；A2（全局→局部）与 A2b（替专家断言）基本不命中，主要靠人工
   发现。需要新增两类检测（本阶段不做）。
2. **D2 页面展示的是记录里的旧 `evidence_check`**：b2 #32 的控制器误报会原样显示，
   页面没有"当前检查器重算结果不同"的标注。重建页面只重算指纹，不重算检查。
3. **D3 分类字段为空**：27 条 revise 的 `category` 全为 `null`；导入/校验允许
   分类为空，导致无法按类别自动汇总。是否强制 revise/reject 必须分类，待用户决定。

**本阶段不做**：不改 b2 记录/trace/审核 JSON/台账；不自动改判；不冻结 b2；不修改
提示词、停止逻辑、证据解释规则；不启动外部 Agent 200 会话或 SFT。

### 4.17 专家组重选：候选核实与评测计划（2026-10-18 用户定稿）

**目标**：保留 PROBE 作为基线，优先核实并独立评测 SPAI、AIDE、DRCT；SSP 第二梯队，
DIRE 作为较高成本的研究对照。最终只保留 **2–3 个可靠且互补**的专家；找不到合格的
第二专家就如实说明，不为凑数保留无效专家。

**硬约束**：

* 先独立评测，不接入 Agent；可增加离线适配器与评测入口，但**不修改**现有 Agent
  提示词、停止逻辑或证据解释规则。
* 与 PROBE 使用**同一批输入**；按生成器、格式/处理条件分别报告。
* 报告 AUROC、官方阈值下真假两类表现、失败数、耗时、显存；不平均原始分数、不把
  分数解释成校准概率；**不根据结果翻转方向或调阈值**。
* MNW + DIV2K 已用于分析；若继续用于选专家，明确记为**开发参考集**；另留未参与
  选择的最终评测源，同源变体不跨分区。

#### 4.17a 候选核实表（版本、权重、许可、数据、预处理、输出、成本）

| 候选 | 版本（代码/论文） | 权重来源与身份 | 许可 | 训练数据 | 输入/测量范围 | 输出/默认阈值 | 成本（估计） | 关键限制 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **PROBE**（基线） | 已在本机；权重 sha256 `c8caa9ba…a450156` | ModelScope `shuinishaojiu/PROBE-AIGI-Detection`，`DINOv2_best_model_step_34999.pth` 1.22 GB | 上游未声明（不再分发） | 上游未声明 | pad→336，非重叠滑窗均值，**global** | sigmoid，高=假，阈值 0.5 | 中位 210 ms/图；外部集 AUROC 0.99–1.0 | 上游语料未知；外部错误仅 13/200 |
| **SPAI** | repo `8ff7b3b6`（2025-08-01），CVPR 2025；代码已固定到 `third_party/spai/repo` | **已上传**：`third_party/spai/weights/spai.pth`，**934,865,338 B，sha256 `24159f27…9648a55`**（云侧计算；官方无公开哈希可比） | Apache-2.0（代码与权重） | 实图自监督：latent-diffusion 训练集、COCO、LSUN（MFM ViT-B/16 预训练；仅训练需要） | 原分辨率，pad 到 ≥224，不缩放/不裁剪；归一化 [0,1] | sigmoid，高=假，阈值 0.5 | 实测 reserved **8.303 GB**、alloc 6.6 GB；稳定 **0.423 s/图**、首次 0.763 s、加载 2.835 s；139.9M 参数 | 适配器已实现；CPU 覆盖 324/324；官方 yaml 与 list 前向路径已核验；加载需 config 补默认项 + `weights_only=False`（yacs CfgNode） |
| **AIDE** | repo `6725b710`（2025-06-04），ICLR 2025；代码已固定到 `third_party/aide/repo` | **三份已上传**：`progan_train`（主）**3,592,077,976 B，sha256 `ce3a9d66…a4a756e`**；`sd14_train` **3,592,077,976 B，sha256 `382a50d1…2f91989`**；`GenImage_train` **3,592,100,460 B，sha256 `f084346d…9a9ff16`**；**主干已含在 checkpoint**（`openclip_convnext_xxl.*` 378 个张量，897.9M 参数），无需另下 ConvNeXt | 代码 MIT；Chameleon 数据集仅学术；checkpoint 许可未单独声明 | CNNSpot（ProGAN）+ GenImage | 4 个 DCT 重建变体 + 原图，Resize 256×256，ImageNet 归一化 | 2 类 softmax，`[:,1]`=假，阈值 0.5 | 897.9M 参数；实测 reserved **3.709 GB**、alloc 3.545 GB；稳定 **1.256 s/图**、首次 1.596 s、加载 4.730 s | **内部分支算一个专家**；主候选固定 `progan_train`，另外两份保留、不混用；适配器已实现，CPU 覆盖 1025/1025 |
| **DRCT** | repo `01aa7d0b`（2025-04-05），ICML 2024 | ModelScope `BokingChen/DRCT-2M` `pretrained.zip`：**4,178,790,241 B，sha256 `235c006947cb87a44425afc1480f74f0a25713095686a6d0c10d3495c08dd046`** | 仓库**无 LICENSE**；权重许可未声明 | DRCT-2M（MSCOCO 实图 + SD inpainting/v1-4 重建）/ GenImage | 测试 pad + **CenterCrop 224**（`--is_crop`），ImageNet 归一化；**center crop** | 2 类 softmax，`[:,1]`=假，阈值 0.5 | ConvB ~0.05 s/图，显存 4–6 GB；CLIP ViT-L 略高 | **必须区分主干/训练集/DR 版本**；README 更正 ConvB 在 BigGAN 仅 59.81%；推理**不需要**重建（测试脚本不传 `--is_dire`） |
| **SSP**（第二梯队） | repo `38e9ab30`（2026-03-03），arXiv 2402.01123 | 8 个按 GenImage 生成器训练的 checkpoint，Baidu/Dropbox；**Dropbox 不可达、Baidu 需客户端；大小/sha256 待下载** | MIT（代码） | GenImage 8 个生成器（每数据集一个 checkpoint） | 单 patch（`patch_size=32`）→ Resize 256，SRM + ResNet50 | sigmoid，高=假，阈值 0.5 | ~0.05 s/图，显存 ~3 GB | **SSP ≠ ESSP**（ESSP 未发布，不能套用抗压缩结果）；无统一 checkpoint，选权重会与评测生成器重叠；`test.sh` 的 `--patchsize` 是笔误 |
| **DIRE**（研究对照） | repo `1f89fff1`（2024-09-26），ICCV 2023 | 分类器 checkpoint 在 RecDrive/Baidu/OneDrive（RecDrive 可达但需密码/API；sha256 待下载）；扩散模型 `256x256_diffusion_uncond.pt` **2,211,383,297 B**（OpenAI Azure，可及） | 仓库根无 LICENSE；guided-diffusion 为 MIT；权重许可未声明 | DiffusionForensics（LSUN/ImageNet/CelebA-HQ，ADM） | DDIM 反演+重建（20 步，256），`abs(img-recon)` → Resize 256/CenterCrop 224/ImageNet | sigmoid，高=假，阈值 0.5 | 每图数秒（重建）+ 分类器；显存 8–12 GB | **重建误差图 ≠ 篡改定位图**；`demo.py` 直接喂原图，与论文/`compute_dire.py` 路径不一致；扩散专用，对 GAN 假图可能弱 |

**权重与代码状态（2026-10-26 更新）**：`/root/autodl-tmp` 余 **16 GB**，overlay
余 2.1 GB。SPAI/AIDE 四份权重已由用户上传并完成**云侧 sha256**：
`experts/candidate_weights.json`（注意：官方只提供 Google Drive，无公开哈希，
不能写成"官方哈希已验证"）。代码已按 commit 固定：SPAI `8ff7b3b6`（70 个源文件）、
AIDE `6725b710`（19 个源文件），每文件哈希在各自 `SOURCE.json`；项目内
`experts/candidate_code.json` 记录 commit 与清单哈希。

**仍待下载**：

| 对象 | 链接 | 放置位置 | 备注 |
| --- | --- | --- | --- |
| DRCT `pretrained.zip` | [ModelScope `BokingChen/DRCT-2M`](https://modelscope.cn/datasets/BokingChen/DRCT-2M) | `/root/autodl-tmp/third_party/drct/weights/pretrained.zip` | **4,178,790,241 B，sha256 `235c0069…08dd046`**；ModelScope 本机可达，可由我下载；主配置取 ConvB + DRCT-2M + SDv1（`14_acc0.9996.pth`），解压后只保留所需 checkpoint |
| SSP 8 个 checkpoint | [Baidu（密码 bcmi）](https://pan.baidu.com/s/1BTkl1gc7gHq4Ev457G__QQ?pwd=bcmi) / [Dropbox](https://www.dropbox.com/scl/fo/2xdvtew4rjrrsl6cseq30/AElVHGO84W1DSmlImMg1ruM?rlkey=e1a2hnzh62wuuxrnbkfv6f90v&st=getws7lp&dl=0) | `/root/autodl-tmp/third_party/ssp/weights/` | 第二梯队；本机 Dropbox 不可达、Baidu 需客户端；8 个按生成器训练的 checkpoint，无统一权重，无公开哈希 |
| DIRE 分类器 checkpoint | [RecDrive（密码 dire）](https://rec.ustc.edu.cn/share/ec980150-4615-11ee-be0a-eb822f25e070) / [Baidu（密码 dire）](https://pan.baidu.com/s/1Rdzc7l8P0RrJft0cW0a4Gg) | `/root/autodl-tmp/third_party/dire/weights/` | 研究对照；扩散权重 `256x256_diffusion_uncond.pt`（2,211,383,297 B）可从 OpenAI Azure 下载，无需你上传 |
| （不需要）AIDE ConvNeXt 主干 | — | — | checkpoint 已含 378 个主干张量 |
| （不需要）SPAI MFM ViT-B/16 预训练 | — | — | 仅训练需要；推理 checkpoint 已含训练后权重 |

下载/上传后先核对大小与 sha256，再进正式评测；未核对的权重不进入报告。

#### 4.17b 开发评测集（首轮 200 输入；最终留出封存）

* **预注册分区（先于任何候选评测，已落盘）**：MNW 清单每组有 250 张源图，当前只用
  了 20 张。已按源 sha256 + 固定 seed 写入 `dataset/mnw_div2k/dev_final_split.json`：
  每组 **当前开发 20 / 扩展开发 20 / 最终留出 20**（5 组共 300 源；任一分区之间
  **没有共享 sha256**，同一源的所有格式变体只能落在同一分区）。最终留出从**未参与
  开发**的源中抽取；这是"同生成器、不同源图"的留出，**不是**未见生成器；若需要
  未见生成器，另找数据集并单列。
* **D1a（首轮开发集，唯一评测口径）**：现有 **100 源 × 2 个条件**（original、
  jpeg_q95）= **200 个输入/专家**，直接使用现有
  `dataset/mnw_div2k/inputs_manifest.json`（清单本身即 200 条、100 源）。q95 与现有
  清单使用**同一编码器/参数**，原图不动、不缩放。同一源的 2 个变体全部留在开发集。
* **q85/q70 暂缓**：现有清单未生成 q85/q70；首轮**不新增、不补跑**，不参与选专家。
  等首轮独立检测与描述性错误重合复核后，再单独决定是否补做；届时应先写清输入数、
  目的和预算，**不把首轮隐式扩成 400 输入**。
* **不再设自动 D1b 扩样**：首轮若 PROBE 错误数不足或某些组 CI 过宽，记录为
  "证据不足/不可判定"，不自动追加开发源；任何扩样都要另立步骤并重新授权。
* **最终留出集（封存）**：从最终分区取 **每组 20 源 ×2 条件**（original +
  jpeg_q95）；4 个生成器组 + DIV2K 共 100 源 → **200 输入**。所有变体留在最终分区。
  最终集**只在候选短名单确定后跑一次**，不参与选专家；若最终分区不够或需要未见
  生成器，最终评测推迟，不用开发集冒充。
* **独立单位**：首轮是 **100 个独立源**（80 Fake + 20 Real）、**200 个源×条件单元**；
  20 张 DIV2K Real 在四个生成器组间复用，四组不独立。**不能把 200 当 200 个独立源，
  也不能跨组池化 CI。**
* **理由**：首轮目标是把独立检测与同源重复测量算清楚，而不是追求错误样本量。
  PROBE 在现有 200 输入上错误仅 **13/200**（8/100 源至少错一次）；首轮因此不设
  "最少纠正数"门槛，q85/q70 留到有明确需要时再评估。

#### 4.17c 指标与互补性分析（预注册）

1. **独立检测（每专家、每组 × 条件）**：
   * 组内配对 **20 Fake + 20 Real**（DIV2K）；AUROC + 95% bootstrap CI（重采样源）；
   * 官方默认阈值（0.5）下 Fake 召回、Real 召回（TNR）、balanced accuracy；
   * **失败数单列**；耗时分开记录**模型加载 / 首次推理 / 稳定推理**（注明是否含
     预处理、数据传输、CUDA 同步）；host 峰值 RSS、cgroup current、GPU
     allocated/reserved 峰值；
   * 保留**每图** raw score、predicted label、true label，以及模型身份（权重
     sha256、代码 commit、预处理、输出语义）。
   * **独立源计数**：首轮 = **100 个独立源**（80 Fake + 20 Real）、**200 个
     源×条件单元**；独立单位仍是 **100 个源**。组内配对 n=40 源；20 张 DIV2K Real
     在四个生成器组间复用，**四组不独立**；不把 200 当独立样本，不跨组池化 CI。
2. **互补性（描述性 2×2；不是准入）**：`PROBE 错 & 候选对`（纠正）、
   `PROBE 对 & 候选错`（损害）、`都对`、`都错`。
   * 按组、按条件分别报告四格计数、**sample ID** 与**每格不同源数**；同一源的
     original/q95 是**重复测量**，不能当 2 个独立事件。
   * **净纠正 = 纠正 − 损害**只作描述；CI 用**按源聚类 bootstrap**（1000 次，
     固定 seed 20261026；重采样源并携带其全部条件）。DIV2K 的 20 张 Real 在四组
     复用，组间不独立；只做组内解释，不把四组汇总成单一数字。
   * **取消硬门槛**：不设“净纠正 > 0”“至少 8 个纠正单元”“AUROC ≥0.70”
     “整体必须胜过 PROBE”或任何自动准入规则。候选纠正部分 PROBE 错误只说明
     **互补潜力**，不据此声称 Agent 级提升；候选总体弱于 PROBE 也不直接淘汰。
   * **不平均原始分数、不做晚期融合、不把候选分数当校准概率、不翻转方向、
     不调阈值**。
3. **专家组决定（人工、事后）**：首轮结束后，根据独立检测表现、失败率、成本与
   描述性互补潜力综合选择 **2–3 个专家**（或只保留 PROBE）；理由、反例和证据不足
   之处写清楚，不做自动阈值判定，不为凑数保留无效专家。
4. **方向/阈值**：全部使用各官方仓库的默认方向与阈值（当前均为高=假、0.5）；
   **不因外部结果翻转方向、不调阈值**。
5. **失败处理**：权重不可用/依赖缺失/加载失败/显存不足记为
   `not_evaluated(reason)`；单图异常或输出非有限记 `failed`，**不当 0/中性分、
   不当错误预测、不插补**；失败过多的专家在结论中单列。

#### 4.17d 执行顺序与资源预算

| 步骤 | 内容 | 产物/门槛 | 资源 |
| --- | --- | --- | --- |
| 0 | 权重/代码/参数覆盖与官方预处理核对 | **已完成 SPAI/AIDE**（sha256/覆盖/预处理/代码 pin）；DRCT/SSP/DIRE 仍未下载 | 无 GPU |
| 1 | 离线适配器 + CPU 测试，不改 Agent | **已实现** `experts/spai.py` / `experts/aide.py`；`core` 721 通过 | CPU |
| 2 | **官方路径对照**：SPAI 官方 CLI / AIDE 官方 `TestDataset`，同图张量、logit、分数、重复调用 | **已完成，均 `ok=true`**（见 §4.17e） | 已用 GPU（核验） |
| 3 | **首轮开发集 D1a：每专家 200 输入**（100 源 × original/q95）；PROBE、SPAI、AIDE **各自独立进程、顺序执行** | 逐图分数 + 分组报告 + 失败/耗时/内存 | **待授权**；实测估计 **<20 min** |
| 4 | 描述性 2×2 错误重合（按源聚类）；**不设合格线**；人工确定 2–3 个专家 | 候选短名单 + 结论 + 反例 | CPU/分析 |
| 5 | 最终留出集（封存 100 源 ×2 条件）复评入选专家 | 最终报告；只跑一次 | 待短名单；估计 <20 min |
| 6 | 之后才结合 §4.16 做协议/证据包/管线修订与新小批验收 | 新小批 + 人工验收 | 另计 |

**实际命令（每专家一个进程，统一使用 `dataset/mnw_div2k/inputs_manifest.json`）：**

```bash
# PROBE（主环境；GPU 需授权）
FORENSIC_ALLOW_GPU=1 python scripts/g8_external_eval.py \
    --inputs dataset/mnw_div2k/inputs_manifest.json --experts probe

# SPAI（候选 venv；独立进程）
FORENSIC_ALLOW_GPU=1 /root/autodl-tmp/venvs/candidates/bin/python \
    scripts/g8_external_eval.py \
    --inputs dataset/mnw_div2k/inputs_manifest.json --experts spai

# AIDE progan_train（候选 venv；独立进程；不与 SPAI 同进程）
FORENSIC_ALLOW_GPU=1 /root/autodl-tmp/venvs/candidates/bin/python \
    scripts/g8_external_eval.py \
    --inputs dataset/mnw_div2k/inputs_manifest.json --experts aide

# 描述性错误重合（CPU；每个候选单独生成）
python scripts/g8_candidate_compare.py \
    --probe sft_data/g8_external_eval/<probe_run>/scores.jsonl \
    --candidate sft_data/g8_external_eval/<spai_or_aide_run>/scores.jsonl \
    --out sft_data/g8_external_eval/compare_<candidate>.json
```

**预算（以 2026-10-26 实测为准）**：

| 专家 | 加载 | 首次 | 稳定中位 | 200 输入估算 | GPU 峰值 alloc/reserved | host maxRSS | cgroup current（核验时） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PROBE | 待本轮记录 | 待本轮记录 | 历史中位 **0.210 s** | **~0.7–1.4 min** | 待本轮记录 | 待本轮记录 | 待本轮记录 |
| SPAI（官方 list 路径） | **2.835 s** | **0.763 s** | **0.423 s** | **~1.4–2.8 min** | **6.600 / 8.303 GB** | **1.489 GB** | **2.744 GB** |
| AIDE（progan_train） | **4.730 s** | **1.596 s** | **1.256 s** | **~4.2–8.4 min** | **3.545 / 3.709 GB** | **4.407 GB**（含 mmap） | **2.848 GB** |

* 200 输入 × 稳定中位是下限；真实数据分辨率不一，按 1.5–2× 余量估。三专家合计
  **<20 min**（不含排错/重跑）；DRCT/SSP/DIRE 未下载，不在本轮预算内。
* 测量边界：稳定耗时含该专家 `score()` 的预处理与同步；首次含 CUDA 初始化/首次
  kernel；加载单独计时。host maxRSS 是进程峰值（mmap 页计入），cgroup current 是
  进程结束前后读数，`memory.peak` 未暴露。
* **本阶段不启动外部 Agent 200 会话，不生成 SFT 数据，不修改 Agent 提示词/停止
  策略/证据解释规则，不改 b2 原始记录。**

#### 4.17e 官方路径核验、内存实测与准备状态（2026-10-26）

**CPU 检查（无 GPU、无前向）：**

| 项目 | 结果 |
| --- | --- |
| 权重文件 | SPAI `934,865,338 B` / `24159f27…`；AIDE `progan_train` `3,592,077,976 B` / `ce3a9d66…`；`sd14_train` 同大小 / `382a50d1…`；`GenImage_train` `3,592,100,460 B` / `f084346d…`。**大小与 SHA256 均与 `experts/candidate_weights.json` 一致**（云侧计算；官方无公开哈希，不能写成“官方哈希已验证”） |
| 代码版本 | SPAI `8ff7b3b6`（70 个源文件）、AIDE `6725b710`（19 个源文件）；每文件哈希在 `third_party/*/repo/SOURCE.json`，汇总在 `experts/candidate_code.json` |
| SPAI 参数覆盖 | 模型 324 键 / checkpoint 324 键，**missing 0 / unexpected 0 / shape mismatch 0**；`check_state_dict` 对缺失/多余/形状不符**直接抛错**，没有用 `strict=False` 掩盖 |
| AIDE 参数覆盖 | 模型 1025 键 / checkpoint 1025 键，**missing 0 / unexpected 0 / shape mismatch 0**；`openclip_convnext_xxl.*` **378 个张量**，总 **897.9M** 参数（主干已含，无需另下） |
| SPAI 官方预处理 | 332×500 合成图 → `[3,332,500]`、float32、范围 `[0,1]`；官方 `configs/spai.yaml`，`ORIGINAL_RESOLUTION=True`，不缩放/不裁剪；`MINIMUM_PATCHES=4`（checkpoint 内是 1，**以官方 yaml 为准**） |
| AIDE 官方预处理 | 332×500 合成图 → `[5,3,256,256]`、float32、ImageNet 归一化；顺序为 4 个 DCT patch `[3,32,32]` + 原图；官方 `TestDataset` |
| 适配器 | `experts/candidate_common.py`（meta 构建 / 流式加载 / 覆盖检查）、`experts/spai.py`、`experts/aide.py`；`scripts/g8_external_eval.py` 支持 `--experts spai,aide`；**不注册进 Agent 工具箱** |
| 环境 | 独立 venv `/root/autodl-tmp/venvs/candidates`（einops 0.8.0、open-clip-torch 2.24.0、albumentations 1.4.14、albucore 0.0.16、filetype、lmdb；复用系统 torch 2.5.1 / timm 1.0.30）。**主环境未被修改** |

**stub 修复与同进程连续调用**：SPAI 的 seaborn/pandas stub 现在带
`importlib.machinery.ModuleSpec`（loader=None）；`has_module()` 对无 spec 的
`sys.modules` 条目也有回退。主环境连续 3 次 `unavailable_reason()` 稳定报告缺少
`einops/albumentations/filetype/lmdb`；venv 连续 3 次稳定报告可用，**没有再次出现
`find_spec()` 异常**。连续调用验证的是**真实专家接口**（`SPAIExpert.analyze` /
`AIDEExpert.analyze`），不是只调 `unavailable_reason()`。

**官方实现 vs 适配器（同图逐项，`scripts/verify_candidate_official.py`）：**

| 检查项 | SPAI | AIDE（progan_train） |
| --- | --- | --- |
| 官方入口 | `python -m spai infer`（独立进程，官方 CLI + 官方 config + 官方 checkpoint） | 官方 `data.datasets.TestDataset` |
| 预处理张量 | 2/2 图 `torch.equal=True`，max abs diff **0** | 2/2 图 `torch.equal=True`，max abs diff **0** |
| 模型 logits | 适配器 vs 官方 tensor 前向：diff **0**（Fake `0.16683943569660187`；Real `-23.47699546813965`） | 2 类 logits 全等：Fake `[1.4218192100524902, -1.4248559474945068]`；Real `[-0.5993655920028687, 0.5093327760696411]` |
| 最终分数 | 适配器 = 官方 CLI = 重复 analyze：Fake **0.5416133999824524**；Real **6.368987282012739e-11** | softmax[:,1] 适配器 = 官方 = 重复 analyze：Fake **0.05485343933105469**；Real **0.7518863677978516** |
| 输出索引/方向 | sigmoid 标量；高=假；阈值 0.5 | `softmax(logits, dim=1)[:,1]`；类 1=假；阈值 0.5 |
| 覆盖 | 324/324 | 1025/1025（含主干 378） |
| 重复调用 | 同进程两次 `analyze` 全等、有限 | 同进程两次 `analyze` 全等、有限 |
| 官方 CLI 端到端 | 两张图 CSV 分数与适配器**逐位相同**；进程总耗时 15.246 s（含启动/加载） | —（官方 `engine_finetune.evaluate` 的分布式 `all_gather` 循环未调用，见下） |
| GPU 峰值 alloc/reserved | **6.600 / 8.303 GB** | **3.545 / 3.709 GB** |
| host maxRSS / cgroup current | **1.489 GB / 2.744 GB** | **4.407 GB（含 mmap） / 2.848 GB** |

* **SPAI 的关键接线修正**：官方 `validate` 把**每张图一个 `1×3×H×W` 张量的 list** 传给
  `forward_arbitrary_resolution_batch`；适配器早期传堆叠张量会走 `forward_batch`，同一图
  分数不同（`0.559819` vs 官方 `0.541613`）。现在 `raw_logits` 固定传 list +
  `FEATURE_EXTRACTION_BATCH`，并按官方 `configs/spai.yaml` 构建。**堆叠张量路径已废弃。**
* **AIDE 的核验边界**：模型构建与 mmap 流式加载复用适配器（checkpoint 已含 ConvNeXt
  主干），官方 `engine_finetune.evaluate` 的分布式循环未调用；但**官方 `TestDataset`
  预处理与官方输出公式是独立调用的**，同图张量/logit/分数全等。结论只覆盖“预处理 +
  前向 + 输出索引”，不覆盖官方分布式评测封装。
* RGB/BGR：官方数据集读取均为 PIL RGB；`Expert.analyze` 接收 BGR 后转换，重复调用与
  官方分数一致，说明入口转换未改变数值。resize/crop/pad、归一化、patch 选择与顺序、
  模型 config、输出索引均由上述张量全等/分数全等逐项覆盖；**没有用“假图分数应高于
  真图”判断接线**。上表两张图的分数只作逐位对照记录，不构成性能结论——AIDE 的示例
  Fake 分数低于 Real 恰好说明不能用两张图判方向，方向由官方类定义与 `[:,1]` 决定。
* 身份绑定：报告写入权重 sha256、代码 repo/commit、预处理、输出语义与阈值
  （`_expert_identity`）；权重路径与哈希在 `experts/candidate_weights.json`，代码 pin 在
  `experts/candidate_code.json`；全部输出做有限性检查，加载失败/依赖缺失/异常输出/OOM
  记为失败，不写 0/中性分。
* **计时口径差异**：官方核验 JSON 里 SPAI 的 `first_analyze_s=3.036 s` 是该进程在官方
  CLI 之后另起 CUDA 上下文/模型 warmup 的首次调用；用于预算的是同一官方路径 smoke 的
  加载 2.835 s、首次 0.763 s、稳定中位 0.423 s。两者不矛盾，但**不能混用**。

**内存与 cgroup 实测（2026-10-26）**：

| 项目 | 实测值 | 说明 |
| --- | --- | --- |
| cgroup 路径 | `/proc/self/cgroup` → `0::/`；文件在 `/sys/fs/cgroup/memory.*` | 当前可直接读取 |
| `memory.max` | **128,849,018,880 B = 120 GiB** | 不是 2 GB |
| `memory.high` | **126,701,535,232 B ≈ 118 GiB** | 软限制 |
| `memory.current` | 空闲 **1,794,301,952 B ≈ 1.67 GiB** | 核验时 SPAI 2.744 GB / AIDE 2.848 GB |
| `memory.peak` | **不存在**（读取 ENOENT） | 不能用它取历史峰值 |
| `memory.events` | low/high/max/oom/oom_kill **全 0** | 两次核验无 OOM |
| `memory.stat` | anon ≈ 1.00 GB、file ≈ 0.69 GB、slab ≈ 0.075 GB（空闲态） | `file` 含页缓存，不能当进程匿名内存 |
| host | 总内存 ≈ **1.0 TiB**，available ≈ **940 GiB**；ulimit memory/virtual 均 unlimited | 主机层面充裕 |
| 进程 RSS | SPAI **1.489 GB**；AIDE **4.407 GB** | `ru_maxrss` 含 mmap 的 checkpoint 页（file-backed），不等于匿名内存 |

* **与早先“2 GB”读数的关系**：2026-10-26 早些时候 CPU 检查读到
  `memory.max=2,147,483,648`（2 GiB），合并两项覆盖检查曾以 **exit 137** 结束；当前同一
  路径已读到 120 GiB，`memory.events` 无 OOM 记录。**无法从现有记录判断当时是临时限制、
  另一个 cgroup 还是读数时机差异**，因此只把 2 GB 记为“当时观测”，不再作为当前预算；
  也不把 exit 137 追认为代码缺陷。后续仍**一次只跑一个候选进程**，便于峰值归因和释放。
* **测量口径**：`score()` 包含预处理、H2D 传输、前向与 `.item()` 同步；加载单独计时；
  首次包含 CUDA 初始化/首次 kernel。GPU 峰值取 `max_memory_allocated/reserved`；host
  峰值取 `ru_maxrss`；cgroup 取进程前后 `memory.current`，不声称是峰值。
* **预算更新**：SPAI 稳定 0.423 s/图、AIDE 1.256 s/图；200 输入首轮 SPAI ~1.5–2.5 min、
  AIDE ~4.2–5 min，PROBE 历史中位 0.210 s/图 → ~0.7 min，加加载/重试 **<20 min**。
  显存 SPAI reserved 8.303 GB、AIDE 3.709 GB，均远低于 24 GB 卡；两进程结束后
  `nvidia-smi` 回到 1 MiB，无残留进程。

**准备状态**：`core` 全绿 **721 passed**（含 `test_candidate_adapters.py`、扩展后的
`test_external_eval.py`、`test_candidate_compare.py`）；权重/代码/覆盖/官方路径/重复调用/
内存均已核实。**未运行开发集，未启动外部 Agent，未生成 SFT 数据；未改 Agent 提示词、
停止策略、证据解释规则或 b2 原始记录。** 等用户明确授权后，按 §4.17d 的命令顺序执行
首轮开发集。

### 4.18 旧第三方模型清理（2026-10-26，用户确认）

**决定**：用户确认 Qwen2.5-VL-7B-Instruct 已退出、后续不再使用，只删除这一份旧权重；
TruFor 等其余第三方资产按用户选择**保留**。

**删除对象**（唯一）：

* `/root/autodl-tmp/psychology_video_project/models/models/qwen--Qwen2.5-VL-7B-Instruct.PENDING-DELETE-20261015`
  —— 17 个文件、5 个 safetensors、约 16 GB。该目录自 2026-10-15 起已改名为
  `PENDING-DELETE`，本次是删除确认，不是新发生的隔离。

**删除前核实**：无进程占用、项目内无符号链接指向它；`config.QWEN_MODEL_PATH` 指向的
旧名字在改名后已不存在；另一项目 `psychology_video_project` 的硬编码路径自 2026-10-15
起已悬空（其最后提交 2026-08-17）。删除只影响历史复算/旧项目 A3，不影响 g8 的
Qwen3-VL 主链。

**未删除（明确保留）**：`third_party/TruFor`（593 MB）、`TruFor_weights.zip`
（249 MB）、`trufor_test.py`；`third_party/{probe,npr,univfd,spai,aide}`；项目内
`experts/trufor.py` 及其适配器/测试；`noise`/`jpeg`/`frequency_v2` 等内部实现。

**后果**：`--mllm qwen25` 在缺权重时显式报错，历史复算需重新下载约 16.6 GB；
`scripts/train_lora_g4.py` 仍依赖旧基座，不能直接作为 Qwen3 SFT 入口；
`psychology_video_project` 若恢复 A3 复核，需重新下载权重并改回路径。
**磁盘**：`/root/autodl-tmp` 可用从 312 MB 增至 **16 GB**（overlay 余 2.1 GB）。

### 4.5 小批 Agent 人工验收（test 之前）

- 小批（先 8–16 源 × 4 格式）跑真实 Qwen3-VL + 新工具箱，产出**最小审阅视图**：原始图像、实际投递的证据包、
  完整原始对话、结案与外部控制原因。
- 人工抽查项：视觉分析是否具体、证据引用是否忠实（含字段与方向）、重复请求处置、停止原因可追溯、
  三类回复分别留存。
- 通过后才考虑打开 test（§3c 的前置三件套 + 本项）。

**b1 小批结果（2026-10-15，8 源 × 4 格式 = 32 会话，真实 Qwen3-VL + probe/npr/univfd）**：

| 格式 | n | 判定正确 | 弃权 | 错 |
| --- | --- | --- | --- | --- |
| png / jpeg_q95 / jpeg_q85 / jpeg_q70 | 各 8 | 各 8 | 0 | 0 |

- 32/32 `model_concluded`（无预算耗尽、无循环、无格式失败）；引用检查 32 个 pass、0 缺陷；投递形状全部 `slim`；0 次失败工具；均 27.5 s/会话，峰值显存 18.1 GB。
- **但 32 个会话全部只调用了 `probe_dino_expert`，npr/univfd 一次都没被调用**（每条都是"1 次调用"）。这与 g7 批次的形状一致：PROBE 是主判据，提示词也明说"其他专家只是佐证、可选"，模型就照办了。**后果**：这批数据没有在**模型闭环**里检验 NPR/UnivFD（我另行单独验过它们的真机路径）；若按此扩大生成，得到的仍是"PROBE 主导"的数据。是否要调整提示词/协议让模型更愿意取第二意见（或明确接受 probe-only），需用户定。
- 两条候选发现（`text_value_as_measurement`、`direction_conflict`）逐句核过：**都是账本启发式的假阳性**（"低于阈值 0.5"引用的阈值本来就在包里；"未被识别为 AI 生成"是否定句被读成断言）。已修 + 加回归测试；b1 记录保留原样（记录是那次运行的产物，不回溯改写）。
- `admit_trajectories_g4`（**训练**准入，不是验收门）对本批 32/32 判 reject：理由全是旧世界（`unknown tool 'npr_expert'` ×32、要求无工具基线）。验收视图因此不走准入（`scripts/g8_review_workstation.py` 直读记录 + trace，见 §4.6）。准入要对齐新工具箱属阶段 5 的活。

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

## 9. 外部评测数据准备（2026-10-16，用户授权下载）

- 下载到 `dataset/mnw_div2k/`：MNW 的 Adobe_Firefly_v4、Google_Imagen4、ByteDance_Seedream_v4、Midjourney_v7 全部图片（各250张），以及 DIV2K Validation HR（100张）。
- MNW 固定提交 `42b0188b580bc955d54b26f2ea7016021fc708e7`；原图保持文件名和字节，按 LFS SHA256/大小核验，DIV2K 验证 ZIP CRC 和图片解码。保存来源和文件哈希清单。
- 图片与下载缓存不入 Git；不修改现有数据分区、不调用模型/GPU、不生成 SFT。MNW 仅供非商业评测，DIV2K 仅供学术研究；这是一份组合外部数据，不声称逐图绝对未见，测试时另行固定抽样并分析格式/来源差异。

### 9.1 工程规范资产描述同步（2026-10-16）

按用户要求同步 `docs/development/agent.md` §4：当前Qwen3模型身份、固定revision、实测加载路径/精度/显存与历史Qwen2.5隔离状态；新增本机MNW/DIV2K资产数量、路径、清单、用途限制。不得把本机下载写成已上传AutoDL，不改代码或实际资产。

### 9.2 提交前状态核对（2026-10-16）

1100张图片仍在本机；此前生成的manifest、下载清单、目录README及ZIP包已不在目录内。§9的保存清单为下载时计划/动作，不代表现在仍有该资产。同步README与agent资产表述，保留历史操作记录，不重建清单、不下载、不删除图片；仅提交下载脚本、忽略规则与文档，图片不入库。后续评测前另行固定样本及其哈希清单。

### 9.3 清单入库与只读核对（2026-10-17）

`dataset/mnw_div2k/manifest.json`（`schema: external_dataset/1`，1100 条：path/group/label/bytes/sha256/url）已提交入库；图片仍按 `.gitignore` 不入库。提交前只读核对：清单 ↔ 磁盘**双向一一对应**（无缺图、无清单外文件）；随机 40 张的 sha256 与字节数 **40/40** 一致；清单内**无重复 SHA256**；`images_reencoded: false`。

**用途限制（随清单一起入库，不得被后续便利覆盖）**：`purpose = External evaluation only; do not use for SFT or threshold fitting`；MNW `Evaluation only; no training or commercial use`；DIV2K `Academic research only`。清单自带的 caveats 同样生效：组合集非官方配对测试集、**不保证**与骨干预训练数据逐图不相交、格式/分辨率/内容混杂必须分开分析。

因此：这份清单解锁的是**外部评测**（仍需 GPU 与固定抽样规则），**不解锁训练**——训练线仍走 b2 验收与协议验证。
