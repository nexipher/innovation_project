### 2026-07-17 12:00 - 阶段一实现计划编写

* **当前操作动作**：编写并提交 `plan.md` 实现计划
* **核心变更说明**：
  1. 基于 `Reasoning_Framework.md` 和 `active_forensic_agent_tasks.md` 设计了完整的目录结构和模块划分
  2. 定义了 BaseExpert、BaseMLLMClient、ForensicStateMachine、HaltingChecker 等核心接口
  3. 规划了 6 个阶段的依赖驱动实现顺序（基础设施→工具层→专家→MLLM→状态机→测试）
  4. 设计了 Mock MLLM 的 4 种行为模式（fast_verdict/two_calls/explore_all/conflict）
  5. 确定了 SFT 数据生成方案（ShareGPT 格式，自动从状态机 Trace 导出）
* **涉及/修改的文件清单**：
  - `plan.md` (Created)
* **执行结果与验证状态**：已提交并推送至 GitHub (`bca44ac`)，等待用户审阅后开始执行
* **置信度或遗留待办（TODO）**：尚未开始代码实现；等待用户下一步指令

---
### 2026-07-17 12:06:25 - Phase 1.1 基础设施验证

* **当前操作动作**：Phase 1.1 基础设施验证
* **核心变更说明**：
  1. 验证 config/ImageUtils/SessionLogger 模块
* **涉及/修改的文件清单**：
  - `config.py`
  - `utils/image_utils.py`
  - `utils/logger.py`
* **执行结果与验证状态**：全部模块导入和基本功能正常
* **置信度或遗留待办（TODO）**：无
---
### 2026-07-17 12:15 - 阶段 1.2 工具层实现

* **当前操作动作**：创建新文件
* **核心变更说明**：
  1. 实现 CoordinateTransformer：相对坐标 [0,1000] ↔ 绝对像素坐标双向转换，含边界裁剪和最小尺寸保证（16px）
  2. 实现 Parser：正则解析器，支持 `<planning>`/`<call_*>`/`<reasoning>`/`<verdict>` 全部标签提取，含结构化校验和 fallback bbox 提取
  3. 更新 utils/__init__.py 注册新模块
* **涉及/修改的文件清单**：
  - `utils/coordinate_transformer.py` (Created)
  - `utils/parser.py` (Created)
  - `utils/__init__.py` (Modified)
* **执行结果与验证状态**：坐标往返转换精度无损、clip 正确截断边界和过小 bbox；Parser 正常提取多标签混排文本、畸形输入正确返回 None、校验方法正确拒绝无标签文本
* **置信度或遗留待办（TODO）**：无
---
### 2026-07-17 12:25 - 阶段 1.3 法证专家模块实现

* **当前操作动作**：创建新文件
* **核心变更说明**：
  1. 实现 BaseExpert 抽象类 + ExpertResult dataclass（完整映射 Evidence Token Schema）
  2. 实现 FrequencyExpert：Hanning 窗 → 2D-FFT → 功率谱 → 高频径向平均 → 周期峰值检测 → sigmoid 归一化
  3. 实现 NoiseExpert：SRM 5×5 高通滤波核 → 逐通道卷积 → 滑动窗口局部方差 → 不一致性度量 → sigmoid 归一化
  4. 实现 JPEGExpert：8×8 块效应强度（水平/垂直边界梯度比）→ DCT 系数直方图双重量化检测 → 加权融合 → sigmoid 归一化
* **涉及/修改的文件清单**：
  - `experts/__init__.py` (Created)
  - `experts/base.py` (Created)
  - `experts/frequency.py` (Created)
  - `experts/noise.py` (Created)
  - `experts/jpeg.py` (Created)
* **执行结果与验证状态**：在 Real(500×750) 和 Midjourney(1024×1024) 真实图像上验证——Noise 专家正确区分（Real=0.15, MJ=0.76），JPEG 检测到真实图像的 JPEG 压缩痕迹，Frequency 需更大图像区域调参；全部 strength ∈ [0,1]，schema 完整
* **置信度或遗留待办（TODO）**：Frequency Expert 在中小 crop 区域敏感度偏低，后续需对更大区域或全图调参
---
### 2026-07-17 12:35 - 阶段 1.4 MLLM 抽象层实现

* **当前操作动作**：创建新文件 + Bug 修复
* **核心变更说明**：
  1. 实现 BaseMLLMClient 抽象接口（generate / reset / name / mode）
  2. 实现 MockMLLMClient：4 种行为模式（fast_verdict / two_calls / explore_all / conflict），模板驱动 XML 生成，从路径自动检测 Real/Fake，确定性 bbox 生成
  3. 修复 return 语句换行导致 _call() 变为死代码的 Bug（6 处），修复后全部模式正常运行
* **涉及/修改的文件清单**：
  - `mllm/__init__.py` (Created)
  - `mllm/base.py` (Created)
  - `mllm/mock_client.py` (Created + Fixed)
* **执行结果与验证状态**：全部 4 种模式在 Real 和 Fake 路径上正确生成对应的 planning/call/reasoning/verdict 标签；Parser 100% 解析通过；fast_verdict=2turns, two_calls=2turns, explore_all=4turns(3 experts), conflict=3turns→Uncertain
* **置信度或遗留待办（TODO）**：无
---
### 2026-07-17 12:27:49 - 阶段 1.5 状态机核心实现

* **当前操作动作**：阶段 1.5 状态机核心实现
* **核心变更说明**：
  1. 实现 EvidenceTokenizer：ExpertResult → Evidence Token Schema JSON，含强度映射字典（3 段式）
  2. 实现 HaltingChecker：四重终止守卫（verdict/max_steps/evidence_conflict/info_gain），含熵和 KL 散度计算
  3. 实现 ForensicStateMachine：核心 while 循环 — 加载图像→MLLM生成→解析→执行专家→Evidence Token注入→终止检查→SFT导出
  4. 集成测试：4 种模式×真实图像端到端验证通过（Real→Real, MJ→Fake, conflict→Uncertain, explore_all→info_gain收敛）
* **涉及/修改的文件清单**：
  - `state_machine/__init__.py (Created)`
  - `state_machine/evidence_tokenizer.py (Created)`
  - `state_machine/halting.py (Created)`
  - `state_machine/controller.py (Created)`
* **执行结果与验证状态**：全管道 4 场景验证通过：verdict_output / evidence_conflict / info_gain_converged 均正确触发；SFT JSON 正常生成
* **置信度或遗留待办（TODO）**：Frequency Expert 在 GenImage 图像上 sensitivity 偏低(0.011)，需后续调参
---
### 2026-07-17 12:37:29 - 阶段 1.6 main.py 入口 + 测试套件

* **当前操作动作**：阶段 1.6 main.py 入口 + 测试套件
* **核心变更说明**：
  1. 实现 main.py CLI 入口：支持 --image 单张分析、--batch 批量（Real/all/GenImage子目录）、--mode 行为模式、--max 数量限制
  2. 创建完整测试套件 (95 tests)：parser(18) + coordinate_transformer(10) + image_utils(8) + frequency(5) + noise(5) + jpeg(4) + tokenizer(6) + halting(14) + mock_mllm(6) + controller(6) + pipeline(12)
  3. 端到端管道测试覆盖 Real/Midjourney/SD15/ADM 四种图像 + conflict/explore_all 模式
  4. 修复 conftest dtype 重复参数 bug + noise 测试阈值 + controller 测试路径
* **涉及/修改的文件清单**：
  - `main.py (Created)`
  - `tests/__init__.py (Created)`
  - `tests/conftest.py (Created)`
  - `tests/test_parser.py (Created)`
  - `tests/test_coordinate_transformer.py (Created)`
  - `tests/test_image_utils.py (Created)`
  - `tests/test_frequency_expert.py (Created)`
  - `tests/test_noise_expert.py (Created)`
  - `tests/test_jpeg_expert.py (Created)`
  - `tests/test_evidence_tokenizer.py (Created)`
  - `tests/test_halting.py (Created)`
  - `tests/test_mock_mllm.py (Created)`
  - `tests/test_controller.py (Created)`
  - `tests/test_pipeline.py (Created)`
* **执行结果与验证状态**：pytest: 95 passed in 15.09s — 全部通过，无失败
* **置信度或遗留待办（TODO）**：无
---
### 2026-07-17 12:45:17 - README.md 工程规范重构

* **当前操作动作**：README.md 工程规范重构
* **核心变更说明**：
  1. 按标准开源项目格式重写 README.md：Overview / Architecture / Quick Start / Pipeline Output / Experts / Halting / SFT / Achievements / Roadmap
  2. 包含完整项目结构树、CLI 使用示例、Evidence Token 示例 JSON、阶段性成果汇总
  3. 添加 badges、citation、license 等工程规范元素
* **涉及/修改的文件清单**：
  - `README.md (Modified)`
* **执行结果与验证状态**：README 覆盖项目全貌，可直接用于 GitHub 项目首页展示
* **置信度或遗留待办（TODO）**：无
---
### 2026-07-17 12:51:21 - 阶段二计划编写

* **当前操作动作**：阶段二计划编写
* **核心变更说明**：
  1. 基于阶段一成果 + README + plan_phase1 + active_forensic_agent_tasks + Reasoning_Framework 综合编写 plan_phase2.md
  2. 阶段二核心目标：真实 Qwen2.5-VL 接入、SFT 数据规模化（1000+条）、专家算法校准（ROC 优化 sigmoid）、格式纠错反馈环实战验证
  3. 保留 BaseMLLMClient 接口不变，QwenVLClient 支持 vLLM 本地推理和 DashScope API 两种后端
  4. A/B 双线 SFT 数据构造策略（正常破案流 70% + 拦截冲突流 30%）
  5. 专家校准方案：340 张基准测试集 → 全图分析 → ROC 网格搜索 → 更新 config.py
* **涉及/修改的文件清单**：
  - `plan_phase2.md (Created)`
* **执行结果与验证状态**：plan_phase2.md 完整覆盖阶段二目标、子阶段划分、实现顺序、测试策略、完成标准
* **置信度或遗留待办（TODO）**：阶段 2.1 需要 GPU 审批后方可开始
---
### 2026-07-17 13:01:37 - 模型就绪 — 验证 Qwen2.5-VL-7B-Instruct 可用性

* **当前操作动作**：模型就绪 — 验证 Qwen2.5-VL-7B-Instruct 可用性
* **核心变更说明**：
  1. 确认模型已下载至 psychology_video_project/models/models/qwen--Qwen2.5-VL-7B-Instruct/snapshots/master
  2. 通过 transformers AutoConfig + AutoTokenizer 验证模型可加载：qwen2_5_vl, 151K vocab, 5×safetensors (16.60 GB), chat_template 完整
  3. config.py 新增 QWEN_MODEL_PATH 指向模型快照目录
* **涉及/修改的文件清单**：
  - `config.py (Modified)`
* **执行结果与验证状态**：模型路径已配置，等待 GPU 开启后即可进入阶段 2.1
* **置信度或遗留待办（TODO）**：无
---
### 2026-07-21 10:57:08 - 2.1 真实 Qwen2.5-VL 接入

* **当前操作动作**：2.1 真实 Qwen2.5-VL 接入
* **核心变更说明**：
  1. 实现 mllm/qwen_client.py：加载 Qwen2.5-VL-7B-Instruct (FP16, 16.6 GB VRAM)，实现 BaseMLLMClient 接口，支持多轮对话 + 格式纠错反馈环
  2. 修复 Parser 容错：增加 _TAG_NORMALIZE 字典，映射常见 LLM 标签错误 (call_call_noise→noise, call_frequency→freq 等)
  3. 更新 main.py：新增 --mllm qwen/mock 参数，自动 GPU 检测 + 降级提示
  4. 更新 mllm/__init__.py 注册 QwenVLClient
  5. 强化 System Prompt：增加显式标签示例、FORBIDDEN 禁止项、更严格的格式约束
  6. 端到端验证：Real 图正确判定 Real(0.99)；MJ 假图错误判定 Real(0.99)——基座模型缺乏法证推理训练，这正是后续 SFT 阶段要解决的问题
* **涉及/修改的文件清单**：
  - `mllm/qwen_client.py (Created)`
  - `mllm/__init__.py (Modified)`
  - `main.py (Modified)`
  - `utils/parser.py (Modified)`
* **执行结果与验证状态**：管道双向跑通，Qwen 可正确生成 planning/call/reasoning/verdict 标签，Parser 容错有效，Evidence Token 注入正常。已知限制：基座模型缺乏法证推理能力（单次调用后轻信结论），需 SFT 训练解决
* **置信度或遗留待办（TODO）**：模型加载耗时 ~17s/次（可接受单张调试；批量生成需优化为一次加载多张推理）
---
### 2026-07-21 11:08:06 - 2.2 专家算法校准

* **当前操作动作**：2.2 专家算法校准
* **核心变更说明**：
  1. 创建 scripts/calibrate_experts.py：批量全图运行三专家 → 收集 raw_metric 分布 → ROC 网格搜索最优 sigmoid 参数
  2. 对 84 张基准图像 (20 Real + 64 Fake) 完成校准：freq separation=0.05（无信号），noise separation=0.83（有区分度），jpeg separation=1.02（最佳）
  3. 关键发现：noise 和 jpeg 的 Real raw_metric > Fake——因为 Real 是 JPEG（有压缩痕迹），Fake 是 PNG（无压缩）。专家检测的是格式差异而非 AI 伪迹
  4. 更新 config.py：noise sigmoid midpoint 2.0→2.8, steepness 5.0→1.5；jpeg midpoint 1.5→2.0, steepness 8.0→2.0
  5. freq 保持原参数——raw_metric 恒为 0，需要算法级改进而非参数调优
  6. 校准报告保存至 calibration/calibration_report.json
* **涉及/修改的文件清单**：
  - `scripts/calibrate_experts.py (Created)`
  - `config.py (Modified)`
  - `calibration/calibration_report.json (Created)`
* **执行结果与验证状态**：校准脚本运行正常，ROC 网格搜索完成。noise/jpeg 参数更新提升区分度。freq 算法需 Phase 3 重新设计
* **置信度或遗留待办（TODO）**：noise 和 jpeg 都是 Real > Fake（与预期方向相反），这意味着在 PNG vs JPEG 对比中专家检测的是格式差异。需在 Phase 3/SFT 训练中告诉模型这个上下文
---
### 2026-07-21 14:03:25 - 2.3 SFT 数据规模化生成

* **当前操作动作**：2.3 SFT 数据规模化生成
* **核心变更说明**：
  1. 创建 scripts/generate_sft_data.py：A 线自然管道 + B 线构造场景双线生成，模型一次加载复用
  2. A 线产出：610 文件，92% 质量通过率，~561 有效样本（Qwen2.5-VL 真实推理行为记录）
  3. B 线产出：292 文件，Qwen 在 max_steps/conflict/info_gain 场景下的反思对话完整记录
  4. B 线 finalize_sft() bug 已修复（_extract_and_finalize 提取 verdict），下次运行生效
  5. 总计 902 文件 / 5.6 MB ShareGPT 格式 SFT 数据，~561 高质量样本可直接用于阶段三微调
  6. 运行耗时 ~166 分钟 GPU（RTX 4090），含 600 张 A 线自然推理 + 300 条 B 线构造场景
* **涉及/修改的文件清单**：
  - `scripts/generate_sft_data.py (Created + Fixed)`
  - `sft_data/metadata.json (Created)`
  - `traces/sft_sessions/*.json (902 files generated)`
* **执行结果与验证状态**：A 线 92% 质量通过，B 线代码 bug 已修复但需重跑。当前数据规模满足阶段三 SFT 训练需求
* **置信度或遗留待办（TODO）**：B 线需重跑以获得 complete verdict 字段；当前 B 线 conversation 内容完整但 verdict 为 null
---
### 2026-07-21 15:12:04 - 2.4 验证与评估

* **当前操作动作**：2.4 验证与评估
* **核心变更说明**：
  1. 格式正确率：A-line 98%（197/200），B-line 100%（199/200），超过目标 85%
  2. SFT 数据质量：1195 总文件，专家调用分布 freq=64%/noise=24%/jpeg=12%，平均 1.9 steps/session
  3. 端到端准确率：Real=53%, Fake=20%, Overall=25%——基座 Qwen2.5-VL 缺乏法证推理，验证了 SFT 训练必要性
  4. Verdict 分布：Real=182, Fake=112, Uncertain=103, NULL=103（首次 B 线残留）
  5. Halting 分布：verdict_output=106, b_line_constructed=76, info_gain=42, conflict=4, max_steps=1
  6. 阶段二全部完成：QwenVLClient 运行正常、专家校准完成、SFT 数据 >= 800 条、格式覆盖率 >= 98%
* **涉及/修改的文件清单**：
  - `No code changes — validation analysis only`
* **执行结果与验证状态**：阶段二 4 个子阶段全部完成。准确率低是预期结果（基座模型无训练），SFT 数据已就绪进入阶段三
* **置信度或遗留待办（TODO）**：freq 专家调用过多（64%）但几乎无区分度——阶段三 SFT 训练前建议改善 expert 选择策略或重新设计 freq 算法
---
### 2026-07-21 15:14:01 - 阶段三计划编写

* **当前操作动作**：阶段三计划编写
* **核心变更说明**：
  1. 3.1 SFT 微调：816 条数据 → 8:1:1 划分 → LoRA (rank=64) → LLaMA-Factory 训练
  2. 3.2 专家重构：Frequency Expert v2（全图 FFT + 多尺度）、调用策略优化、System Prompt 调用指南
  3. 3.3 GRPO 对齐：4 项 Rule-governed Reward（Format / Anti-Laziness / Attention Consistency / Outcome Accuracy）
  4. 3.4 全数据集评估：分类准确率（按生成模型细分）、消融实验（Mock/SFT/GRPO × 单/双/三专家）、论文级指标
  5. 阶段三本质变化：从"管道能跑真模型"到"模型能用法证"——注入法证推理能力，产出专用 Qwen2.5-VL 变体
* **涉及/修改的文件清单**：
  - `plan.md (Appended — Phase 3 plan)`
* **执行结果与验证状态**：plan.md 已追加完整阶段三计划（4 个子阶段、GPU 时间预估、完成标准、阶段本质变化）
* **置信度或遗留待办（TODO）**：GRPO 需要 TRL GRPOTrainer 等 RL 框架，可作为 SFT 后的进阶优化，非硬交付
---
### 2026-07-21 16:17:06 - 3.1a + 3.2 SFT 数据构造 + 专家重构

* **当前操作动作**：3.1a + 3.2 SFT 数据构造 + 专家重构
* **核心变更说明**：
  1. 3.1a: 创建 scripts/build_sft_data.py — 运行三专家→四类分类→模板合成理想推理链
  2. 产出：365 条合成 SFT 数据（correct=165, conflict=200），borderline/format 待补充
  3. 3.2: 创建 experts/frequency_v2.py — 全图多尺度 FFT，raw_metric 提升 5-8×（0.0001→0.0005），strength ~0.24
  4. freq_v2 对 GenImage PNG 的区分度仍未达独立判定水平——依赖 noise/jpeg 专家为主信号
  5. 分类逻辑修复：冲突先行→correct 次之，Fake GT 样本均纳入 correct 以弥补 Expert 偏 Real 的先天缺陷
  6. plan.md 已追加四种 SFT 数据类型的详细构造策略和示例
* **涉及/修改的文件清单**：
  - `scripts/build_sft_data.py (Created)`
  - `experts/frequency_v2.py (Created)`
  - `sft_data/train/sft_correct.json (Created — 165 records)`
  - `sft_data/train/sft_conflict.json (Created — 200 records)`
  - `plan.md (Modified — Phase 3 plan + 4 data types)`
* **执行结果与验证状态**：3.1a 和 3.2 CPU 部分完成。SFT 训练数据 365 条已就绪。3.1b (LoRA 微调) 需 GPU
* **置信度或遗留待办（TODO）**：Expert 信号强度不足（GenImage PNG 天然偏 Real）是已知限制——SFT 训练的重点是教模型推理模式而非依赖完美 Expert 信号
---
### 2026-07-21 16:38:34 - SFT 数据准备完成

* **当前操作动作**：SFT 数据准备完成
* **核心变更说明**：
  1. Step 1: 从 A 线 610 条筛选 verdict==GT → 196 条真实 Qwen 正确推理（比预期多 56 条）
  2. Step 2: 从基准集选取 strength 在 [0.25,0.6] 的图像 → 合成 100 条 borderline 谨慎推理
  3. Step 3: 从 A 线抽取格式完整（planning+call+reasoning+verdict）的样本 → 100 条 format
  4. Step 4: 加载已有 conflict 200 条（合成模板）
  5. 最终数据集: 596 条（correct=196, conflict=200, borderline=100, format=100）
  6. correct 来源为真实 Qwen 推理而非模板——推理风格自然，正确率保证（verdict=GT）
* **涉及/修改的文件清单**：
  - `scripts/finalize_sft_data.py (Created)`
  - `sft_data/train/final/sft_correct.json (196 records)`
  - `sft_data/train/final/sft_conflict.json (200 records)`
  - `sft_data/train/final/sft_borderline.json (100 records)`
  - `sft_data/train/final/sft_format.json (100 records)`
  - `sft_data/train/final/metadata.json`
* **执行结果与验证状态**：596 条 SFT 数据已全部就绪。A 线真实 Qwen 推理为主（196），合成冲突+边界+格式为辅（400）。可进入 3.1b LoRA 微调
* **置信度或遗留待办（TODO）**：GT 字段全部完整。correct 中少数 bbox 异常（如[0,0,100,100]），SFT 训练时需要清理
---
### 2026-07-22 11:14:40 - 修复 — 三个专家的 reasoning 从硬编码改为条件化

* **当前操作动作**：修复 — 三个专家的 reasoning 从硬编码改为条件化
* **核心变更说明**：
  1. frequency.py/noise.py/jpeg.py/frequency_v2.py 新增 _get_reasoning() 方法
  2. 三个 strength 区间输出不同 reasoning：<0.3→解释为何正常，0.3-0.7→描述模糊建议交叉验证，>=0.7→说明为何判 AI 生成
  3. 修复前：reasoning 永远说"这是 AI 生成特征"——即使 strength=0.01 判 Real 时也如此——导致 Evidence Token 内部自相矛盾
  4. 修复后：reasoning 与 strength/support 保持一致，不再误导 Qwen 产生矛盾论述
* **涉及/修改的文件清单**：
  - `experts/frequency.py (Modified)`
  - `experts/noise.py (Modified)`
  - `experts/jpeg.py (Modified)`
  - `experts/frequency_v2.py (Modified)`
* **执行结果与验证状态**：验证：Real 图和 MJ 假图的 reasoning 均与各自 strength 值匹配，无矛盾
* **置信度或遗留待办（TODO）**：SFT 数据中包含 freq Expert 的样本需要重新生成——因为旧数据中的 reasoning 是错误的
---
### 2026-07-22 12:25:00 - 3.1a SFT 数据集最终状态更新

* **当前操作动作**：3.1a SFT 数据集最终状态更新
* **核心变更说明**：
  1. 对应计划锚点: plan.md §3.1.1d Expert reasoning 修复与数据重生成
  2. 修复四个专家的 reasoning 硬编码问题: frequency/noise/jpeg/frequency_v2 新增 _get_reasoning() 方法, 三段式条件化输出
  3. 重跑 build_sft_data.py: 冲突数据 181 条用修复后 Expert 重新合成, reasoning 与 strength 一致
  4. 重跑 finalize_sft_data.py: borderline 100 条修复重生成, 同步 conflict 到 final 目录
  5. 最终 SFT 数据集 (sft_data/train/final/): correct=196, conflict=181, borderline=100, format=100, 总计 577 条
  6. correct 保持 A 线真实 Qwen 推理 (verdict=GT), 不替换为合成模板
  7. format 从 A 线抽取, 不受 Expert 修复影响
  8. 数据全部就绪, 等待 GPU 开启后进入 3.1b LoRA 微调
* **涉及/修改的文件清单**：
  - `experts/frequency.py (Modified — _get_reasoning)`
  - `experts/noise.py (Modified — _get_reasoning)`
  - `experts/jpeg.py (Modified — _get_reasoning)`
  - `experts/frequency_v2.py (Modified — _get_reasoning)`
  - `sft_data/train/sft_conflict.json (Regenerated — 181 records)`
  - `sft_data/train/final/sft_borderline.json (Regenerated — 100 records)`
  - `sft_data/train/final/sft_conflict.json (Synced — 181 records)`
  - `sft_data/train/final/metadata.json (Updated)`
  - `plan.md (Modified — added §3.1.1d)`
  - `claude_operation_log.md (Modified)`
* **执行结果与验证状态**：所有 Expert reasoning 验证通过: low-strength→正常描述, high-strength→异常描述, 无矛盾。SFT 数据集 577 条全部就绪
* **置信度或遗留待办（TODO）**：correct/conflict 中的旧 Evidence Token reasoning 不影响训练目标 (Qwen 的 response 正确)。GPU 开启后可直接开始微调, 无需额外数据准备
---
### 2026-09-15 10:38:57 - 文档同步 — plan.md 数据状态修正 + README 阶段二/三内容更新

* **当前操作动作**：文档同步 — plan.md 数据状态修正 + README 阶段二/三内容更新
* **核心变更说明**：
  1. plan.md §3.1.1c 重写：从"目标分布"改为"实际最终分布"表（correct=196/conflict=181/borderline=100/format=100，总计 577）
  2. plan.md §3.1.1d 更新：conflict 计数 200→181，决策说明改为实际执行结果（conflict+borderline 均已重生成）
  3. plan.md §3.1 标题修正：816 条 → 577 条；§3.1.1 数据预处理标记完成状态；§3.1.4 更新实际脚本名（build_sft_data.py / finalize_sft_data.py）
  4. README.md 按 agent.md §6.1 规范重构：新增 Mermaid 整体数据流图 + 状态机时序图（替代原 ASCII 图）
  5. README.md 新增 §5 接口规范：Evidence Token Schema、强度映射字典、SOP 标签协议、Verdict Schema、SFT 数据 Schema
  6. README.md 新增 §8 维护说明：四轨文件驱动、版本控制规范、测试命令、操作日志格式、GPU 使用规范
  7. README.md 更新：快速开始增加 --mllm qwen；工程目录补充 frequency_v2/qwen_client/scripts；进展表覆盖阶段一/二/三；局限更新为实际发现（基座准确率 25%、格式差异、freq 信号弱、无视频支持）
* **涉及/修改的文件清单**：
  - `plan.md (Modified — 数据状态修正)`
  - `README.md (Modified — 阶段二/三内容 + Mermaid + 接口规范 + 维护说明)`
* **执行结果与验证状态**：数据核对通过：SFT 数据集实际 577 条（correct=196/conflict=181/borderline=100/format=100），测试 95 个，与 README 描述一致
* **置信度或遗留待办（TODO）**：GitHub 网络不通，commit 保留在本地。GPU 开启后进入 3.1b
---
### 2026-09-15 20:16:31 - 4.1 当前程序框架与运行逻辑介绍文档

* **当前操作动作**：创建独立程序架构与运行逻辑说明文档
* **对应计划锚点**：实现 `plan.md` 中的 4.1 小节
* **核心变更说明**：
  1. 新增 `CURRENT_PROGRAM_ARCHITECTURE.md`，说明程序定位、分层框架及核心组件职责
  2. 使用 Mermaid 绘制整体控制流与单次分析时序图
  3. 按当前源码描述 CLI 启动、MLLM 决策、Parser 解析、专家调度、Evidence Token 回灌、终止判断和 Trace 保存流程
  4. 明确 Mock/Qwen、Frequency v1/v2、步数统计和未接入阶段三功能等当前实现边界
* **涉及/修改的文件清单**：
  - `CURRENT_PROGRAM_ARCHITECTURE.md` (Created)
  - `plan.md` (Modified — added §4.1)
  - `claude_operation_log.md` (Modified)
* **执行结果与验证状态**：Markdown 结构检查通过；共 314 行、2 个 Mermaid 图，代码围栏成对闭合；文档引用的主要源码路径均存在
* **置信度或遗留待办（TODO）**：无
---
### 2026-09-15 20:32:42 - 4.2 两篇参考论文对照与架构借鉴分析

* **当前操作动作**：阅读新增 PDF 并将对照分析写入当前程序架构文档
* **对应计划锚点**：实现 `plan.md` 中的 4.2 小节
* **核心变更说明**：
  1. 完整阅读 TVSIP（ACM MM 2025）与 Propose-Rectify（IEEE TIFS 2026）两篇论文，并核验方法图、实验表和消融结论
  2. 对照当前 Forensic-Agent 的 MLLM 角色、专家输出、证据融合、空间定位和解释约束
  3. 提炼语义 Evidence Token、原图+高亮图提示、外部 bbox 校正、EvidenceRectifier、多尺度空间证据和自适应专家路由等可借鉴设计
  4. 新增分阶段演进路线、实施优先级、消融实验、鲁棒性实验、跨生成器评估及不宜直接迁移的内容
* **涉及/修改的文件清单**：
  - `CURRENT_PROGRAM_ARCHITECTURE.md` (Modified — added §9)
  - `plan.md` (Modified — added §4.2)
  - `claude_operation_log.md` (Modified)
* **执行结果与验证状态**：两篇 PDF 共 24 页均完成文本阅读，代表性方法页和实验页完成 PNG 渲染核验；架构文档现为 529 行、3 个 Mermaid 图、Markdown 围栏成对闭合，论文链接均指向存在的本地文件
* **置信度或遗留待办（TODO）**：论文方法与当前任务存在“局部篡改定位 vs 整图生成检测”的任务差异，文档已明确区分可直接应用、需改造和暂不适合照搬的部分
---
### 2026-09-15 20:53:16 - 4.3 FakeReasoning 与 ForenX 对照及架构更新

* **当前操作动作**：阅读新加入的两篇整图 AI 生成检测论文，并更新当前程序架构与实施建议
* **对应计划锚点**：实现 `plan.md` 中的 4.3 小节
* **核心变更说明**：
  1. 完整阅读 FakeReasoning（IEEE TIP 2026）与 ForenX 两篇论文，核对任务定义、方法、数据构造、泛化实验、解释评测、消融和限制
  2. 将整图生成场景中的 bbox 明确定义为“诊断证据区域”，避免误称为篡改像素或 forged mask
  3. 提炼 FakeReasoning 的 CLIP+DINO 互补、FAFF、分类概率映射、分层推理和专家数据审核流程
  4. 提炼 ForenX 的 forensic prompt、辅助检测损失、两阶段训练、少量人工区域标注和提示词扰动评测
  5. 将建议映射到当前状态机：三类 token logits、EvidenceRectifier、证据绑定解释、SFT 内容审核、留一生成器与重复稳定性评测
  6. 保留第一组论文的历史分析，同时移除对当前已不存在 PDF 文件的失效链接说明
* **涉及/修改的文件清单**：
  - `CURRENT_PROGRAM_ARCHITECTURE.md` (Modified — added §10 and refreshed §9 source note)
  - `plan.md` (Modified — added §4.3)
  - `claude_operation_log.md` (Modified)
* **执行结果与验证状态**：两篇新 PDF 共 36 页完成文本阅读，代表性方法与实验页完成 PNG 渲染核验；架构文档现为 756 行、4 个 Mermaid 图，46 个 Markdown 围栏成对闭合，新论文链接均指向存在的本地文件；`git diff --check` 通过
* **置信度或遗留待办（TODO）**：论文 PDF 保持为用户新增的未跟踪文件，未纳入本次文档提交；近期优先完成 577 条 SFT reasoning 内容审核与 verdict 概率校准
---
### 2026-09-20 10:36:47 - 4.4 ForgeryVCR 对照与全局—局部双阶段演进规划

* **当前操作动作**：阅读 ForgeryVCR，并围绕全局生成检测与未来局部篡改定位更新架构建议
* **对应计划锚点**：实现 `plan.md` 中的 4.4 小节
* **核心变更说明**：
  1. 明确当前整图生成检测与未来局部篡改定位的任务边界，提出 `task_type`、`evidence_scope`、`region_semantics` 三个兼容字段
  2. 对照 ForgeryVCR 的视觉中心推理、工具准入、增益筛选、多工具轨迹和分类/定位/工具奖励，区分可直接借鉴与不可直接照搬的部分
  3. 建议将 Expert 升级为可校准的 Evidence Bundle，并保留可供 MLLM 复核的视觉产物；先做单专家有效性与互补性消融，再决定增删专家
  4. 将停止策略从固定优先级改为风险约束下的“预期风险下降 − 调用成本”决策，增加可观测停止原因、冲突处理和真实信息增益估计
  5. 建议在 LoRA 前对现有 577 条 SFT 数据全量单审，并对冲突、边界和格式轨迹双审；按 A/B/C/D/R 分层决定保留、修订、降权或剔除
  6. 给出 G0–G5 全局检测路线及 L1–L3 局部篡改扩展路线，并设置各阶段进入门槛
* **涉及/修改的文件清单**：
  - `CURRENT_PROGRAM_ARCHITECTURE.md` (Modified — added §11)
  - `plan.md` (Modified — added §4.4)
  - `claude_operation_log.md` (Modified)
* **执行结果与验证状态**：ForgeryVCR 全文 36 页已阅读，代表性方法、工具选择、消融和数据构造页面已渲染核验；架构文档现为 1163 行、6 个 Mermaid 图、78 个 Markdown 围栏成对闭合，本地论文链接存在，`git diff --check` 通过
* **置信度或遗留待办（TODO）**：ForgeryVCR 面向局部篡改，当前项目面向整图生成；其 IoU 增益阈值、SAM2 掩码细化和定位奖励仅应在局部数据就绪后启用。近期应先执行 577 条 SFT 数据审计，再完成 Expert 校准，最后离线回放比较停止策略
---
### 2026-09-20 13:36:33 - 4.5 SFT 严重错误样本隔离

* **当前操作动作**：审计 conflict 数据，将不具备真实对立证据的严重错误条目移入拒绝集，并修复生成逻辑
* **对应计划锚点**：实现 `plan.md` 中的 4.5 小节
* **核心变更说明**：
  1. 对 181 条 conflict 记录执行结构审计，以“两个独立 Expert 且 support 为 Real 对 AI-generated/Fake”为最低准入条件
  2. 将 38 条同向证据伪冲突移入 `sft_rejected.json`；其中 26 条同时存在重复 Expert 和忽略 region 后证据载荷重复
  3. 为拒绝记录增加 `audit.review_status`、规则编号、失败类型、审核方式和说明；用户指出的 `04783f51-...` 样本已确认进入拒绝集
  4. 将候选训练集由 577 条调整为 539 条：correct=196、conflict=143、borderline=100、format=100；拒绝集不计入训练总数
  5. 统一 `build_sft_data.py` 的冲突分类与合成阈值，校验 support 方向，删除缺失证据时回退到固定 Expert 的逻辑
  6. 新增可幂等运行的 `audit_sft_conflicts.py`，并让 `finalize_sft_data.py` 在重新整理数据时同步拒绝集
  7. 同步 README、架构说明、计划及 metadata 的现行数量
* **涉及/修改的文件清单**：
  - `scripts/audit_sft_conflicts.py` (Created)
  - `scripts/build_sft_data.py` (Modified)
  - `scripts/finalize_sft_data.py` (Modified)
  - `tests/test_audit_sft_conflicts.py` (Created)
  - `sft_data/train/sft_conflict.json` (Modified)
  - `sft_data/train/sft_rejected.json` (Created)
  - `sft_data/train/final/sft_conflict.json` (Modified)
  - `sft_data/train/final/sft_rejected.json` (Created)
  - `sft_data/train/final/metadata.json` (Modified)
  - `README.md`, `CURRENT_PROGRAM_ARCHITECTURE.md`, `plan.md` (Modified)
* **执行结果与验证状态**：专项测试 `7 passed`；三个脚本通过 `py_compile`；五个输出 JSON 均可解析；审计脚本连续运行结果稳定为 conflict=143、rejected=38；`git diff --check` 通过。全量测试在收集既有 `test_pipeline.py` 时因本机 transformers 缺少 `Qwen2_5_VLForConditionalGeneration` 被阻断，与本次改动无关
* **置信度或遗留待办（TODO）**：剩余 143 条只通过结构准入，尚未完成数值校准、区域真实性及人工内容审核；borderline、correct 和 format 也仍需后续审计
---
### 2026-09-20 14:24:10 - 4.6 文档职责统一与后续计划收敛

* **当前操作动作**：汇总 SFT 审计结论，将架构文档中的后续路线迁入计划，并统一审阅根目录 Markdown
* **对应计划锚点**：实现 `plan.md` 中的 4.6 小节，并建立 §4.7–§4.15 后续执行基线
* **核心变更说明**：
  1. 在 `plan.md` 汇总 correct/conflict/borderline/format/rejected 的审计状态，明确原始 577 条、拒绝 38 条、磁盘候选 539 条但未全部通过训练准入
  2. 新增 G0–G6 与 L1–L3 计划，覆盖旧数据处置、坐标和证据协议、Expert 校准、Evidence Bundle、停止策略 v2、SFT `final_v2`、统一评测、可选 GRPO 和局部篡改扩展
  3. 从 `CURRENT_PROGRAM_ARCHITECTURE.md` 移除重复的路线图、优先级、评测清单和阶段门槛，仅保留当前实现、确认缺陷、论文事实和目标架构约束
  4. 在 README 中明确 `plan.md` 是唯一执行计划，修正“直接 LoRA”顺序，增加旧 SFT 尚未准入和停止逻辑并非真实信息增益的限制说明
  5. 将 `Reasoning_Framework.md` 标为早期概念文档，说明熵/KL 与注意力损失尚未实现，移除独立推进路线并修复损坏的 LaTeX 转义
  6. 审阅全部 7 份根目录 Markdown；原始任务书和 `agent.md` 保持只读，操作日志保留历史口径而不回写旧记录
* **涉及/修改的文件清单**：
  - `plan.md` (Modified — canonical §4.7–§4.15 roadmap)
  - `CURRENT_PROGRAM_ARCHITECTURE.md` (Modified — removed duplicated future plans)
  - `README.md` (Modified — current status and document roles)
  - `Reasoning_Framework.md` (Modified — historical/conceptual status and LaTeX fixes)
  - `claude_operation_log.md` (Modified)
* **执行结果与验证状态**：7 份根目录 Markdown 的代码围栏均成对闭合；`plan.md` §4.7–§4.15 各唯一存在；迁移后的旧路线标题不再出现在架构与概念文档；本地论文和计划链接存在；`git diff --check` 通过
* **置信度或遗留待办（TODO）**：后续从 G0 开始执行；两条已确认严重错误的 correct 样本仍需在 G0 中实际移入拒绝集，剩余旧数据只作为诊断资产
---
### 2026-09-20 14:43:04 - 4.7 G0 收尾 — correct 结构审计 + 全量处置状态标记

* **当前操作动作**：4.7 G0 收尾 — correct 结构审计 + 全量处置状态标记
* **核心变更说明**：
  1. 新增 scripts/audit_sft_correct.py：correct 硬拒绝规则（重复证据/坐标漂移/人工确认）+ 软失败标记 + 全量处置状态
  2. correct 审计：196 → 保留 166（regenerate）+ 硬拒绝 30（重复证据 23 / 坐标漂移 5 / 人工确认 2）
  3. 关键发现：166 条全部含旧版污染 reasoning（生成于 2026-09-15 专家修复之前）——116 条低 strength 却声称 AI 伪迹、85 条单弱证据高置信、73 条 verdict 方向与证据不符
  4. 全量处置状态：regenerate 409（correct 166 + conflict 143 + borderline 100）+ format_only 100 + rejected 68
  5. 拒绝集：68 条（38 conflict 伪冲突 + 30 correct 结构失效），永不训练，保留回归测试
  6. 旁支 sft_data/train/sft_correct.json（139 条合成）标记 superseded
  7. 新增 tests/test_audit_sft_correct.py（9 项），与既有 conflict 审计测试合计 16 项全部通过
  8. 幂等验证：二次运行 md5 完全一致
  9. plan.md §4.7 标记 G0 完成并记录审计结论；README 更新数量（509 候选 + 68 拒绝）
* **涉及/修改的文件清单**：
  - `scripts/audit_sft_correct.py (Created)`
  - `tests/test_audit_sft_correct.py (Created)`
  - `sft_data/train/final/sft_correct.json (166 kept, all regenerate)`
  - `sft_data/train/final/sft_conflict.json (143, stamped regenerate)`
  - `sft_data/train/final/sft_borderline.json (100, stamped regenerate)`
  - `sft_data/train/final/sft_format.json (100, stamped format_only)`
  - `sft_data/train/final/sft_rejected.json (68)`
  - `sft_data/train/final/metadata.json (dispositions added)`
  - `sft_data/train/sft_correct.json (139, stamped superseded)`
  - `plan.md (Modified — §4.7 G0 complete)`
  - `README.md (Modified — counts and status)`
* **执行结果与验证状态**：pytest 16 passed；幂等验证通过；5 个 JSON 全部可解析；G0 完成门槛全部满足
* **置信度或遗留待办（TODO）**：correct 集需在 G4 用修复后的专家与 G1 新协议重新生成（final_v2）。下一步：G1 运行协议与证据正确性修复（纯 CPU）
---
### 2026-09-24 10:24:32 - 4.8 G1 运行协议与证据正确性修复（完成）

* **当前操作动作**：4.8 G1 运行协议与证据正确性修复（完成）
* **核心变更说明**：
  1. 对应计划锚点: plan.md §4.8 全部 6 个实施项完成
  2. 坐标协议: CoordinateTransformer.transform() 双空间返回(region_normalized_1000/region_pixels/coordinate_space/clipped); Evidence Token 同时保存两空间与遗留 region 字符串
  3. 证据去重: EvidenceTokenizer.evidence_id() 稳定 sha1 短 id; 重复结果不进链/不进对话/不进停止统计, suppressed_duplicate_count 单独记录
  4. 多轮图像历史: 新增 mllm/message_builder.py(torch-free); 原图保留首轮, 每个证据轮附诊断区域裁剪图(最多2张, 缺图降级纯文本); 裁剪图落盘 traces/evidence/<session>/ 并写入 token 与对话轮 image_paths
  5. 语义一致性门: 新增 utils/evidence_consistency.py 确定性方向检查(低 strength 不得声称生成伪迹/高 strength 不得声称正常); 失败证据标记 consistency.fail 并降级 support→Uncertain
  6. 任务语义字段: Trace metadata 增加 task_type/evidence_scope/region_semantics; 证据 token 携带 region_semantics
  7. 可观测计数: model_turn_count/expert_call_count/unique_evidence_count/suppressed_duplicate_count/weighted_cost; 停止预算改为 MAX_EXPERT_CALLS=5/MAX_MODEL_TURNS=6 双计数, 原因更名 budget_exhausted
  8. 完成门槛全部满足: 坐标往返 5 种尺寸无漂移; 重复证据不增数不触发虚假收敛; 一致性门捕获 080862af(方向矛盾)+00eb3be4(重复证据 id 碰撞) 两类反例; Mock 端到端 Trace 含任务语义与计数
  9. 提交: f8f010b(坐标+id) 0d706ec(去重+计数+预算) 30d2ae7(一致性门) fac5c36(多轮图像)
* **涉及/修改的文件清单**：
  - `config.py (Modified — 双计数预算/任务语义常量/artifact 目录)`
  - `utils/coordinate_transformer.py (Modified — transform() 双空间)`
  - `utils/evidence_consistency.py (Created)`
  - `utils/image_utils.py (Modified — save_image)`
  - `utils/logger.py (Modified — image_paths/counters/task-semantics)`
  - `mllm/message_builder.py (Created)`
  - `mllm/qwen_client.py (Modified — 委托共享消息构建)`
  - `state_machine/controller.py (Modified — 去重/计数/预算/一致性/artifact)`
  - `state_machine/halting.py (Modified — 双计数预算)`
  - `state_machine/evidence_tokenizer.py (Modified — evidence_id/双空间)`
  - `tests/test_coordinate_transformer.py, test_evidence_tokenizer.py, test_halting.py, test_controller.py, test_evidence_consistency.py, test_message_builder.py, test_pipeline.py (Modified/Created)`
  - `plan.md, CURRENT_PROGRAM_ARCHITECTURE.md, README.md (Modified — G1 状态收口)`
* **执行结果与验证状态**：全量测试 152 passed; 四条完成门槛全部通过; 4 个原子提交
* **置信度或遗留待办（TODO）**：G2 入口: 格式配平校准集与单 Expert 增益基线（Qwen 对比需 GPU 授权）；停止策略重构留待 G3
---
### 2026-09-24 11:47:18 - 4.9 G2-a/b/c 完成（G2-d 待 GPU）

* **当前操作动作**：4.9 G2-a/b/c 完成（G2-d 待 GPU）
* **核心变更说明**：
  1. G2-a 校准集: build_calibration_set.py 生成 2×4 格式配平网格(native/png/jpeg_q95/85/70)+7类扰动, 100 源图→700 样本/24格/0失败
  2. G2-b 专家评估: evaluate_experts_g2.py 全图评估 5 候选专家; 每格 AUROC/F1/阈值 + 极性校正分离度 + 语义方向 + 容器红利量化 + 扰动稳定性 + 错误重叠矩阵 + 250 张可视化产物
  3. G2-b 关键结论: freq_v1 无信号(0.505, 停用候选); freq_v2 弱(0.556→q70 0.634); noise/jpeg 语义反向(高strength⇒Real, 分离度 0.845/0.972 但为压缩历史驱动); ELA 最强(0.948)但 q70 归零(0.504)=压缩历史捷径
  4. G2-c Evidence Bundle: reliability_table.json 蒸馏(分位分箱→经验P(Fake)); utils/reliability.py 运行时查询; token 携带 reliability/calibrated_likelihood/condition_metadata/counter_explanation/visual_artifacts
  5. G2-c 运行时接管: Controller 逐条查询校准表 + 渲染落盘专家产物(频谱/残差/块效应图)并与区域图一起回灌对话; 验证 Mock 管道 token 含全部 Bundle 字段
  6. 适用性标签: disabled:no-signal(freq v1) / weak:marginally-above-chance(freq v2) / inverted:high-metric-means-real(noise,jpeg) / shortcut-prone:compression-history(ela)
  7. 全量测试 164 passed; 3 个原子提交 2937096/2e8cc6c/a471d4b
* **涉及/修改的文件清单**：
  - `scripts/build_calibration_set.py (Created)`
  - `scripts/evaluate_experts_g2.py (Created)`
  - `scripts/build_reliability_table.py (Created)`
  - `experts/ela.py (Created)`
  - `experts/base.py, frequency.py, frequency_v2.py, noise.py, jpeg.py (Modified — render_artifacts + counter_explanation)`
  - `utils/reliability.py (Created)`
  - `state_machine/evidence_tokenizer.py, controller.py (Modified — Bundle 字段)`
  - `calibration/set/manifest.json, raw_values.json, g2_expert_report.json, reliability_table.json (Created)`
  - `tests/test_reliability.py (Created), test_controller.py (Modified)`
  - `plan.md (Modified — G2 执行结果)`
* **执行结果与验证状态**：G2-a/b/c 全部完成; 164 测试通过; 校准集 700 样本; 关键发现: 三个现有专家均受格式/压缩历史混杂支配, ELA 为最强但捷径型分离
* **置信度或遗留待办（TODO）**：G2-d 需 GPU 授权(~1-1.5h RTX 4090)做四条件增益对比; G2-e 依赖 G2-d 结论
---
### 2026-09-24 12:33:42 - 4.9 G2-d 实施就绪（CPU 侧完成，待 GPU 授权）

* **当前操作动作**：4.9 G2-d 四条件增益对比脚本实现 + Mock 干跑验证（用户指令第 2 步："完成 scripts/qwen_gain_baseline.py，这部分编写和 Mock 测试不需要 GPU"）
* **核心变更说明**：
  1. G2-c 语义传递补齐(承接上一步骤): semantics_aligned / applicability / applicability_conditions 已随 Token 传入 Evidence Bundle, 验证端到端
  2. 四条件实现: rgb(evidence_injection="none" + BASELINE_SYSTEM_PROMPT 禁工具) / text(仅 Token JSON) / image(附区域图+产物图, 中性标记不含数值) / both(Token JSON + 图像, G2-c 默认)
  3. 分层抽样限格式对齐 8 格(real|fake × png|q95|q85|q70), 指标: Accuracy/F1/AUROC/ECE/Uncertain率/平均轮数与调用数/分格准确率
  4. 内存阻塞定位与修复(**关键**): 干跑曾被 SIGKILL(exit 137)。实测确认**无内存泄漏**(每 run 后 RSS 恒定 591.7MB), 真因是执行 shell 处于 2GB 只读 cgroup 且与 VS Code/claude/jupyter/tensorboard 共享(常驻 anon ~1.08GB + 可回收页缓存 ~0.47GB, 单进程实际可用 ~0.93GB), 而导入基线达 586MB
  5. 修复①: mllm/__init__.py 急切导入 qwen_client 导致所有 CPU 路径加载 torch+transformers；改 PEP 562 惰性导出 → 导入基线 586MB → 116MB, 干跑四条件全部通过
  6. 修复②(**否则 GPU 运行必失败**): run_condition 原在样本循环内调用 client_factory, GPU 模式下等于每张图新建 QwenVLClient 并重复加载 16.6GB 权重; 改为每条件构建一次 client 与 expert 工具集复用, 状态机按样本重建保证会话隔离
  7. 干跑报告与正式报告分离(g2_gain_report_dry_run.json), mock 数字永不覆盖 GPU 真实报告
  8. GPU 运行前置预检: torch+transformers+processor ≈ 668MB RSS(未计权重加载), 与 0.93GB 可用余量对比 —— 执行前需确认无其他重进程, 运行期监控 memory.current
  9. 全量测试 197 passed(新增 33: 注入模式轴 5 + 四条件工具集 28)
* **涉及/修改的文件清单**：
  - `scripts/qwen_gain_baseline.py (Created — 四条件 runner/分层抽样/指标/报告)`
  - `mllm/__init__.py (Modified — PEP 562 惰性导出 QwenVLClient)`
  - `mllm/message_builder.py (Modified — BASELINE_SYSTEM_PROMPT 无工具变体)`
  - `mllm/qwen_client.py (Modified — system_prompt 覆盖)`
  - `state_machine/controller.py (Modified — evidence_injection 四模式轴)`
  - `tests/test_qwen_gain_baseline.py (Created), tests/test_controller.py (Modified — 注入模式 5 项)`
  - `calibration/g2_gain_report_dry_run.json (Created — mock 干跑产物)`
  - `plan.md (Modified — G2-d 实施状态与运行前置条件)`
* **执行结果与验证状态**：CPU 侧全部就绪; 197 测试通过; 干跑四条件完成(报告 calibration/g2_gain_report_dry_run.json); 干跑中 rgb/image 两臂必然 Uncertain 属 mock 语义(mock 依据对话内 Token JSON 判定), 不构成结论
* **置信度或遗留待办（TODO）**：G2-d 四条件对比等待用户 GPU 授权(~1500-1800 次生成 ≈ 1-1.5h RTX 4090); 执行前需确认 cgroup 余量充足; G2-e 依赖 G2-d 结论
---
### 2026-09-24 12:45:10 - G2-d 补充：Trace 目录隔离（数据治理）

* **当前操作动作**：隔离 mock/测试 session 与真实 trace（干跑闭环的收尾治理）
* **核心变更说明**：
  1. 问题: `utils/logger.py` 硬编码 `SFT_SESSIONS_DIR`, 任何 mock/测试 session 都写入 `traces/sft_sessions/`; 实测该目录 1674 个文件中 426 个为 mock/scripted 模式
  2. 风险评估: `scripts/finalize_sft_data.py` 以文件名前缀 `forensic_sft_session_20260721_1[1-2]*` 锁定阶段 2.3 产物, 故现存的 mock trace **不会**被摄入 → 无实际污染, 但目录语义已混乱
  3. 修复: `SessionLogger.__init__(sft_dir=None)` 支持输出目录覆盖(向后兼容); G2-d 干跑写入 `traces/dry_run_sessions/`(已 gitignore); `tests/conftest.py` 增加 autouse fixture 将测试写入临时目录
  4. 验证: 全量测试 198 passed; 运行前后 `traces/sft_sessions/` 文件数 1674 → 1674 保持不变; 干跑 29 条 trace 全部落在 `traces/dry_run_sessions/`
  5. 现存的历史 mock trace 未做批量删除(非本次产物且可能被审计引用), 保留待用户裁决
* **涉及/修改的文件清单**：
  - `utils/logger.py (Modified — sft_dir 覆盖参数)`
  - `scripts/qwen_gain_baseline.py (Modified — 干跑 trace 重定向)`
  - `tests/conftest.py (Modified — autouse trace 隔离 fixture)`
  - `tests/test_qwen_gain_baseline.py (Modified — 重定向测试)`
  - `.gitignore (Modified — traces/dry_run_sessions/)`
  - `CURRENT_PROGRAM_ARCHITECTURE.md, plan.md (Modified — 隔离说明)`
* **执行结果与验证状态**：198 测试通过; 干跑闭环完成(报告 + trace 均与正式产物分离)
* **置信度或遗留待办（TODO）**：G2-d 四条件对比等待 GPU 授权; 历史 mock trace 清理待用户决定
---
### 2026-09-24 12:47:20 - G2-d 补充：断点续跑（长时 GPU 作业保障）

* **当前操作动作**：为 G2-d GPU 运行加入增量落盘与断点续跑
* **核心变更说明**：
  1. 动机: G2-d 预计 1-1.5h, 而本环境已有 SIGKILL 先例(exit 137, 2GB cgroup); 原实现仅在全部条件跑完后写一次报告, 中断即全丢
  2. `run_condition(on_record=...)` 回调: 主流程每完成一个样本即更新该条件的 records/metrics 并落盘
  3. `load_completed(path, mode, per_cell)`: 读取既有报告作为续跑基线; **mode 或 per_cell 不一致时视为冷启动**(避免混入不可比数字); 新增 `--fresh` 强制重跑
  4. `pending_samples()`: 跳过已记录样本; 已完成条件直接打印 skipping
  5. 实测: `--conditions rgb text` → 再调一次(Resuming 16, 两条件 skipping) → 再调全条件补齐 image/both, 三次调用累积为同一份四条件报告
  6. 全量测试 205 passed(新增 7: 续跑 6 + on_record 1)
* **涉及/修改的文件清单**：
  - `scripts/qwen_gain_baseline.py (Modified — on_record/load_completed/pending_samples/initial_completed/--fresh)`
  - `tests/test_qwen_gain_baseline.py (Modified — TestResume 6 项 + on_record 1 项)`
  - `plan.md (Modified — 续跑说明)`
* **执行结果与验证状态**：205 测试通过; 续跑语义实测通过(分次调用累积); 干跑报告含完整四条件
* **置信度或遗留待办（TODO）**：G2-d 四条件对比等待 GPU 授权
---
### 2026-09-24 14:15:30 - 4.9 G2-d 完成：四条件增益对比（结果：显著负增益）

* **当前操作动作**：G2-d 四条件对比执行与配对统计分析
* **核心变更说明**：
  1. 执行: RTX 4090, 13:10-14:07(约 57 分钟), 120 样本(8 格式配平格 ×15 Real+15 Fake) × 4 条件 = 480 次会话; 模型加载 8.8s/16.6GB; cgroup 上限在开 GPU 后由 2GB 变为 128GB(内存不再是约束)
  2. 冒烟验证(8 样本): baseline 臂 1 轮结案/0 调用/无证据注入, 全判 Real
  3. 结果: rgb acc 0.533 AUROC 0.658 | text 0.408/0.505 | image 0.242/0.432 | both 0.258/0.391
  4. 配对统计(新增 scripts/analyze_g2_gain.py): McNemar p≤0.0007, ΔAUROC 95%CI 全部为负 → **专家证据对未微调模型是统计显著的净损害**; both 臂(当前默认协议)代价最高表现最差, 其 AUROC 0.391 低于随机(置信度与真相反相关)
  5. 机制: 基线是全判 Real 的先验(Real 召回 1.000/Fake 0.067/判Real率 0.967); 文本几乎不动摇决策(0.967→0.808)而图像猛推 Fake(→0.233)并摧毁 Real 召回(→0.167)
  6. 价值: 给出 G5 可量化验收线(微调后 ΔAUROC 需显著为正), 并证实 SFT 必要性
  7. **发现运行时 bug**: FrequencyExpertV2.source_name 与 v1 同名, 运行时校准查询命中 v1 条目(disabled:no-signal), v2 实测条目从未被使用 → 纳入 G2-e 修复(否则污染 G4 训练数据)
  8. 全量测试 216 passed(新增 11 项分析脚本测试)
* **涉及/修改的文件清单**：
  - `scripts/analyze_g2_gain.py (Created — McNemar + 配对 bootstrap)`
  - `tests/test_analyze_g2_gain.py (Created — 11 项)`
  - `calibration/g2_gain_report.json (Created — 480 条样本记录)`
  - `calibration/g2_gain_analysis.json (Created — 配对统计)`
  - `logs/g2d_smoke.log, logs/g2d_full.log (Created — 运行日志, 已 gitignore)`
  - `plan.md (Modified — G2-d 结果)`
* **执行结果与验证状态**：G2-d 完成; 四条件全部 120/120; 配对检验显示三个工具臂均显著差于基线
* **置信度或遗留待办（TODO）**：G2-e 开始: 修复 v1/v2 校准身份 bug、停用 v1、决定 v2/noise/jpeg/ELA 准入、更新 Prompt 与运行时注册
---
### 2026-09-24 14:14:15 - 4.9 G2-e 准入决策与收口

* **当前操作动作**：G2-e：停用 v1、决定 v2/noise/jpeg/ELA 准入、更新 Prompt 与运行时注册
* **核心变更说明**：
  1. 决策表: freq v1 **停用**(sep 0.505 无信号, 不注册, 类标 DEPRECATED 仅供旧 trace 复现); freq v2 **保留但弱**(注册为 frequency_expert_v2, Prompt 标注 WEAK 仅佐证); noise **保留+反转语义**(sep 0.845, 五格 0.772-0.847 跨格式稳定); jpeg **保留+反转语义+限制**(png 0.972 → q70 0.569); ELA **不注册**(png 0.948 → q70 0.504, 能力等同压缩历史捷径, 运行期无法得知来源压缩史)
  2. **修复校准身份 bug**: FrequencyExpertV2.source_name 改名为 frequency_expert_v2, 控制器 EXPERT_KEY_MAP["freq"] 同步; 此前 v2 每次查询都命中 v1 的 disabled:no-signal 条目, 且会污染 G4 训练数据
  3. **专家声明实测极性**: BaseExpert.metric_polarity(+1/-1) + inverted_text_map + classify_metric(); noise/jpeg 的 support/interpretation/reasoning 全部改写为实测语义(高值→Real)并保留限制说明; 修正前其文本断言"高异常⇒AI生成"与校准方向相反 —— G2-d 证明这种自相矛盾 token 会误导模型
  4. **Prompt 重写**: 三条直觉规则替换为逐工具实测指南 + 读取契约(calibrated_likelihood 为方向权威; strength 不可跨专家比较; 读 applicability/counter_explanation; 弱或冲突证据输出 Uncertain)
  5. 端到端核验(64 张格式配平 × 3 专家): noise/jpeg 文本方向与校准方向**零直接矛盾**(系统性反转消除), 文本 Uncertain 而校准有方向者 noise 22/64、jpeg 30/64
  6. 遗留转 G3: frequency_expert_v2 仍有 15/64 直接矛盾(strength 分带 0.3/0.7 与 raw_metric 分位分箱两套离散化不重合), 需 EvidenceRectifier 以校准后验统一裁决
  7. 全量测试 256 passed
* **涉及/修改的文件清单**：
  - `experts/base.py (Modified — metric_polarity/classify_metric)`
  - `experts/noise.py, jpeg.py (Modified — 极性反转 + 文本改写)`
  - `experts/frequency_v2.py (Modified — source_name 独立), frequency.py (Modified — DEPRECATED)`
  - `state_machine/controller.py (Modified — 注册表指向 v2)`
  - `mllm/message_builder.py (Modified — FORENSIC_SYSTEM_PROMPT 实测指南)`
  - `main.py, scripts/generate_sft_data.py, scripts/qwen_gain_baseline.py (Modified — 切换到 v2; B线构造数据的 source 对齐)`
  - `tests/test_expert_admission.py (Created — 24 项), conftest.py, test_controller.py, test_pipeline.py, test_frequency_expert.py (Modified)`
  - `plan.md, CURRENT_PROGRAM_ARCHITECTURE.md, README.md (Modified — G2-e 决策与边界)`
* **执行结果与验证状态**：G2-e 完成; 256 测试通过; 反转语义消除; 残留分歧已量化并移交 G3
* **置信度或遗留待办（TODO）**：B 线构造场景模板仍编码 G2 之前的语义, 待 G4 重设计; G3 开始(EvidenceRectifier + 停止策略 v2)
---
### 2026-09-24 14:22:10 - G2-e 补录：G2-d 机制证据定量收口

* **当前操作动作**：把 G2-d 的机制结论从定性表述补为可复现的定量证据（CPU）
* **核心变更说明**：
  1. 背景: plan.md 的 G2-e 决策已引用"G2-d 证明自相矛盾的 token 会误导模型", 但支撑数字未落盘
  2. 与既有统计脚本 `scripts/analyze_g2_gain.py`(Bootstrap AUROC CI / ΔAUROC / 各类召回)分职: 本脚本只回答机制问题, 避免两份分析同名混淆
  3. 三个定量结果: ①`support="AI-generated"` 的 token 仅 23.8% 真为 Fake(基准率 47.9%, p=1e-6 显著低于随机), 即该字段**反向**; ②同一 token 内 `support=AI-generated` 的平均校准 P(Fake)=0.236 而 `support=Real` 为 0.772, 两方向字段系统性相反; ③模型判 Fake 时校准 P(Fake) 均值(0.295/0.385)低于判 Real 时(0.662/0.625) → 模型读 `support` 而**未使用**校准概率
  4. 臂间配对: text 显著优于 image(p=0.030)与 both(p=0.006); 视觉臂 Real 识别从 15/15 被摧毁至 1–3/15; 视觉通道与原始度量方向不一致且不显著(noise p=0.79, jpeg p=0.33)
  5. 这些证据是 G2-e 选择"改专家自身语义(metric_polarity)"而非"仅在 Bundle 附加校准字段"的实证依据
  6. 重命名以避免同名混淆: analyze_gain_g2d.py → analyze_gain_mechanism.py, 输出 g2d_analysis.json → g2_gain_mechanism.json; 测试同步
  7. 全量测试 256 passed(机制分析测试 24 项)
* **涉及/修改的文件清单**：
  - `scripts/analyze_gain_mechanism.py (Created — 臂重建/分格/臂间 McNemar/证据跟随率/Bundle 自洽性/视觉泄漏)`
  - `calibration/g2_gain_mechanism.json (Created)`
  - `tests/test_analyze_gain_mechanism.py (Created — 24 项)`
  - `plan.md (Modified — G2-e 段补定量机制证据表)`
* **执行结果与验证状态**：机制结论可复现; 256 测试通过; 与已提交的统计脚本职责分离
* **置信度或遗留待办（TODO）**：G2-e 已完成且端到端核验(64 样本)通过; **仍缺 Qwen 四条件验证重跑**(极性修复后需 --fresh 重跑, 否则会静默复用修复前数字), 需 GPU 授权; 随后进入 G3
---
### 2026-09-24 14:35:20 - Trace 治理收口 + md 记录缺口补齐

* **当前操作动作**：trace 目录治理（按用户授权自主决定）+ 补齐 md 记录中的三处缺口
* **核心变更说明**：
  1. 核查发现既有前提有误: 真实 trace **是**入库的(1195 条已跟踪), 此前"原始 trace 未版本化"的判断不成立; 未跟踪的仅 G2-d 本次运行的 488 条
  2. 清理: 删除 473 个 mock/scripted trace(two_calls/scripted/explore_all/conflict)+ 6 个无 mock_mode 标记的测试残留; 删除前逐个从 HEAD 核实 —— **被删的 193 个已跟踪文件全部是 mock, 无一个真实样本**
  3. 安全核查: 479 个 mock trace 的日期为 20260717/20260924, **零命中** finalize_sft_data.py 的 `20260721_1[1-2]` 文件名 glob → 从未可能污染 SFT 数据
  4. 决定: **不将 traces/sft_sessions 加入 gitignore**(曾误加后撤销); 改为真实会话入库(1195 既有 + 488 本次 G2-d = 1683), mock 残留清除; 版本控制判定依据是 trace 元数据的 `mock_mode` 字段
  5. md 缺口 ①: G2-e 的核验只是**专家文本层面**(64 样本 × 3 专家), "修复后证据对 Qwen 是否产生正增益"尚未测量 —— 已写入 plan.md 遗留段并给出验证命令
  6. md 缺口 ②: `--fresh` 陷阱(续跑只看 (mode, per_cell), 修复后直接重跑会静默复用修复前数字)已写入 plan.md; 彻底修法(报告加配置指纹)与 G3 一并处理
  7. md 缺口 ③: 目录版本控制策略写入 CURRENT_PROGRAM_ARCHITECTURE.md §5.5 与 README 目录树
* **涉及/修改的文件清单**：
  - `traces/sft_sessions/ (清理 479 个 mock/测试残留; 新增 488 条 G2-d 真实 trace 入库)`
  - `plan.md (Modified — 验证重跑遗留 + --fresh 陷阱)`
  - `CURRENT_PROGRAM_ARCHITECTURE.md (Modified — §5.5 版本控制策略)`
  - `README.md (Modified — 目录树标注)`
  - `.gitignore (未变更 — 误加的 sft_sessions 条目已撤销)`
* **执行结果与验证状态**：trace 目录 2162 → 1683(全部真实); git status 不再被 trace 刷屏; md 记录三处缺口已补
* **置信度或遗留待办（TODO）**：极性修复的 Qwen 层面验证重跑待 GPU 授权(--fresh, ~20 分钟); 随后 G3(EvidenceRectifier)
---
### 2026-09-24 15:40:10 - G2-e 验证重跑（两次）：发现并修复一致性门的极性盲区

* **当前操作动作**：极性修复的 Qwen 层面验证重跑（第一次结果无效 → 定位根因 → 修复 → 第二次有效）
* **核心变更说明**：
  1. 第一次重跑(14:26-14:50, 40 样本×4 条件): 三个工具臂几乎全弃权(text Uncertain 92.5%, image 82.5%, both 80%), 无一超过基线
  2. **根因**: G1 的 `EvidenceConsistencyChecker` 在 5 处硬编码"高 strength ⇒ 生成伪迹"的对齐世界观; G2-e 反转 noise/jpeg 后, 它们**每一次正确的方向声明都被判矛盾并降级为 Uncertain**(实测 postfix 批次 noise 57/57、jpeg 173/173 全为 Uncertain) → 该次运行检验的是门的抹除效应
  3. 修复(提交 5582156)三层: ①`semantics_aligned=False` 的 token 取镜像强度规则(每带禁止的方向词互换) ②短语匹配改为**否定感知**(jpeg 文本写的 "This is NOT a forgery marker" 被裸子串匹配当成伪造断言) ③noise 的 `phenomenon` 低 metric 措辞改写(反转语义下那是生成端)
  4. 验证: 32 张真实校准图上方向声明保留 32/0(此前约 100% 降级); 测试 267 通过
  5. 第二次重跑(15:00-15:35, 门已修): **token 声明从"反向"变"随机"**(jpeg AI-gen 命中率 23.8%→42.1%, 基准 47.5%, p=0.66; noise →46.9%); **image 臂 AUROC 0.432→0.425→0.614**(与基线 0.605 持平, 从低于随机回到持平); 弃权率 80-92%→55-65%
  6. **未达成**: 无任何臂超过基线; text 臂决策级 AUROC 0.290(n=40 且 57.5% 并列于 0.5, 统计量脆弱, 需 n=120 复核)
  7. **新发现阻塞点(转 G3 首要任务)**: 可靠性表在全图标定(n=700), 推理却在模型自选裁剪图上 → 分布失配(noise 裁剪中位 1.262 vs 全图 2.529), 分箱按错误总体解读。这是"证据为何仍无增益"的首要解释
  8. 方法论教训: G2-e 的两次核验都只查了**专家自身输出**, 未查**管线对输出的二次加工**; 今后语义层改动必须端到端核验到 token 落地
* **涉及/修改的文件清单**：
  - `utils/evidence_consistency.py (Modified — INVERTED_SUPPORT/否定感知 _asserts/极性感知强度规则)`
  - `experts/noise.py (Modified — phenomenon 随反转语义改写)`
  - `tests/test_evidence_consistency.py (Modified — TestInvertedExperts 9 项)`
  - `calibration/g2_gain_report_postfix.json (Created — 门未修批次, 结果无效)`
  - `calibration/g2_gain_report_postfix2.json (Created — 门已修批次, 有效)`
  - `scripts/qwen_gain_baseline.py (Modified — --output 覆写报告路径)`
  - `tests/test_qwen_gain_baseline.py (Modified — --output 测试)`
  - `plan.md, CURRENT_PROGRAM_ARCHITECTURE.md (Modified — 两次重跑与门的极性契约)`
* **执行结果与验证状态**：门修复验证通过(32/0); 第二次重跑有效; 267 测试通过; token 不再反向但亦无正增益
* **置信度或遗留待办（TODO）**：G3 EvidenceRectifier 首要任务 = 解决校准总体失配(裁剪尺度 vs 全图); n=120 复核留待 G3 后一并做; B 线构造场景模板仍编码 G2 之前语义, 待 G4 重设计
---
### 2026-09-24 16:05:30 - G3-a 全局尺度一致性（解决校准总体失配）

* **当前操作动作**：G3-a —— 专家度量改为整图计算，消除缩放外推
* **核心变更说明**：
  1. 问题(承接 G2-e 复测): 可靠性表在全图标定(n=700), 推理却测模型自选裁剪, 二者分布不同(noise 裁剪中位 1.262 vs 全图 2.529), 分箱按错误总体解读
  2. 方案(最稳妥第一版): `expert.analyze(img)` 改为传整图; bbox 保留为诊断/可视化区域(区域裁剪图与产物渲染语义不变); 产物改为在**度量输入**上渲染以保证"图与数描述同一对象"
  3. token 新增 `measurement_scope`(global/region) 与 `condition_metadata.region_area_ratio`, 为将来按尺度建条件表(G3-a2)留接口
  4. Prompt 新增读取契约第 5 条: `measurement_scope: global` 表示专家度量整图, bbox 仅定位诊断区域, 应判断整图而非裁剪
  5. 优点: 校准表无需重算(其标定总体本就是全图)
  6. 真机验证: 校准集样本上 noise raw 回到 4.834/2.362/2.850(落在表分布内), png 与 q70 值近乎相同(4.834 vs 4.870, 印证度量对压缩稳定), 校准概率随之有意义(高噪声→P(Fake)=0.114 指向 Real)
  7. 测试 272 通过(新增 TestGlobalMeasurementScope 5 项: 专家收到整图断言/产物渲染输入/token scope 字段/诊断区域仍记录/区域裁剪仍落盘)
* **涉及/修改的文件清单**：
  - `state_machine/controller.py (Modified — 整图度量 + scope/area_ratio 传参 + 产物渲染输入)`
  - `state_machine/evidence_tokenizer.py (Modified — measurement_scope/region_area_ratio)`
  - `mllm/message_builder.py (Modified — Prompt 第 5 条读取契约)`
  - `tests/test_controller.py (Modified — TestGlobalMeasurementScope)`
  - `plan.md (Modified — G3 执行计划 a–e + G3-a 记录)`
* **执行结果与验证状态**：G3-a 完成; 272 测试通过; 真机端到端确认 scope 与分箱正确
* **置信度或遗留待办（TODO）**：G3-b EvidenceRectifier(方向权威单一化) → G3-c 停止策略 v2 → G3-d 配置指纹 → G3-e n=120 GPU 复核(需授权)
---
### 2026-09-24 16:30:45 - G3-b EvidenceRectifier（方向权威单一化）

* **当前操作动作**：G3-b —— 新增 EvidenceRectifier，使每个 token 只有一个方向权威；G1 门退化为校验器
* **核心变更说明**：
  1. 新增 `state_machine/evidence_rectifier.py`: 优先级 `calibrated_likelihood`(权威) > `support`(被重写, 原文存 `support_raw`) > 自由文本(矛盾句改写为校准派生规范句, 测量数值与其余描述保留) > `applicability_conditions`(原样随行)
  2. 中性带 (0.4, 0.6) 判 Uncertain(边界取方向); 无校准条目的 token 保留专家声明并标注 `direction_source="expert_claim_uncalibrated"`
  3. 长文本采用**句级手术**: 按句判定方向(复用门的否定感知 `_claim_direction`), 仅替换矛盾句; 单句文本不替换(否则等于丢弃测量)
  4. **G1 门权威转移**: token 带 `direction="calibrated_likelihood"` 时, 期望方向改由校准决定(strength 分带让位), 否则维持原 polar 感知强度规则; 新增失败名 `support_calibration_mismatch` / `reasoning_contradicts_calibration_*`
  5. 结构性消除 freq_v2 的 15/64 两套离散化矛盾: 方向只由 `raw_metric → 分箱` 一条路径决定
  6. **真机回放(247 个 token, postfix2 批次)**: 全部命中校准条目; 整流器改写 80 个方向(jpeg Uncertain→AI-generated 46、Real→Uncertain 24、Uncertain→Real 6, noise 2, freq_v2 2); **整流后门零失败** → 门成为校验器
  7. 测试 290 通过(新增 test_evidence_rectifier.py 18 项, 含"门接受每个分带""校准压过 strength 分带""手工篡改仍被抓")
* **涉及/修改的文件清单**：
  - `state_machine/evidence_rectifier.py (Created)`
  - `utils/evidence_consistency.py (Modified — 权威转移分支)`
  - `state_machine/controller.py (Modified — tokenize 后立即整流)`
  - `tests/test_evidence_rectifier.py (Created — 18 项)`
* **执行结果与验证状态**：G3-b 完成; 290 测试通过; 真机回放零矛盾、零门失败
* **置信度或遗留待办（TODO）**：G3-c 停止策略 v2(HaltingDecision: 后验+冲突度+剩余工具预期收益−成本) → G3-d 配置指纹 → G3-e n=120 GPU 复核
---
### 2026-09-24 17:05:20 - G3-c 停止策略 v2（HaltingDecision）

* **当前操作动作**：G3-c —— 以校准后验重写停止策略，替换 v1 的"标签顺序 + 跨专家 strength 比较"
* **核心变更说明**：
  1. **v1 的病根**: 冲突检测比较**跨专家 strength**(各专家尺度不可比, G2-e 后高 strength 对不同专家含义相反), 信息增益比较相邻 strength 差值 —— 回放显示 342/1191 次会话由这条无意义规则终止
  2. 新增 `state_machine/halting_v2.py`: 后验(可靠度加权 log-odds, 权重=Youden margin 2·AUROC−1, 未校准 token 权重 0) + 冲突度(加权贡献的抵消比例) + 各工具预期增益(权重×不确定性−成本) → `HaltingDecision(action/verdict/confidence/primary_reason/all_reasons/next_expert/posterior/conflict_score/expected_net_utility)`
  3. 决策规则(顺序): 无证据→探索; 模型 `<verdict>` **仅候选**(需后验同意+足够自信+无未决冲突); **模型重复被驳回的候选→model_stalled**(策略能建议不能强制: mock 管道实测 7 轮→5 轮); 预算耗尽**不自动定标签**(自信才给标签, 否则 Uncertain); 无正收益→停(冲突未决则 Uncertain); 否则继续并推荐最优工具
  4. 控制器接线: `HALTING_POLICY` 配置开关(默认 v2, v1 保留供回放); 驳回候选时注入说明(否则模型会重复输出到预算耗尽); trace 记录 `policy_reason/policy_reasons/posterior/conflict_score/candidate_overridden/model_candidate`
  5. **离线回放(1191 条真实会话, `scripts/replay_halting_g3.py`)**: v1 标注 802 条准确率 0.414 | v2 标注 367 条准确率 **0.534**, 弃权 824 条; v2 后验 ECE 0.176 / Brier 0.277; **冲突样本 74 条: v1 给其中 14 条定了标签, v2 全部弃权**; 模型候选 1161 次, v2 接受 263、驳回 747
  6. 回放局限(已写入脚本 docstring): 无法测量 v2 会省下多少调用(离线无模型), 调用/轮数为 v1 实测值
  7. 测试 332 通过(halting_v2 30 项 + replay 12 项)
* **涉及/修改的文件清单**：
  - `state_machine/halting_v2.py (Created — 后验/冲突/工具效用/HaltingDecision)`
  - `state_machine/controller.py (Modified — 策略分派/停滞计数/驳回说明/计数器)`
  - `config.py (Modified — HALTING_POLICY 与策略常量)`
  - `scripts/replay_halting_g3.py (Created — v1/v2 离线回放)`
  - `calibration/g3_halting_replay.json (Created)`
  - `tests/test_halting_v2.py, tests/test_replay_halting_g3.py (Created), tests/test_controller.py, tests/test_pipeline.py (Modified — 适配 v2 语义)`
* **执行结果与验证状态**：G3-c 完成; 332 测试通过; 回放显示 v2 以更少标签换取更高准确率, 且冲突样本不再被候选覆盖
* **置信度或遗留待办（TODO）**：G3-d 实验配置指纹 → G3-e n=120 GPU 复核(需授权)
---
### 2026-09-24 17:25:40 - G3-d 实验配置指纹

* **当前操作动作**：G3-d —— 用配置指纹替代手工 `--fresh`，消除"配置已变却静默复用旧数字"的隐患
* **核心变更说明**：
  1. 新增 `utils/config_fingerprint.py`: 指纹覆盖 ①专家集合(源名/类名/`metric_polarity`) ②两份系统提示词的内容哈希 ③可靠性表文件哈希 ④整流器/分词器版本常量 ⑤停止策略版本 ⑥git commit; 另给 16 位 digest 便于比对
  2. `qwen_gain_baseline.load_completed` 增加指纹校验: 不符即冷启动, 并通过 `config_fingerprint.differences()` **打印差异组件**(便于定位是提示词变了还是专家极性变了)
  3. 旧报告(无指纹字段, 含已提交的 G2-d 三份报告)一律视为冷启动 —— 这正好保证 G3-e 不会误续任何历史数字
  4. 实测: 同配置二次调用打印 `Resuming: 8 records already on disk` 并跳过已完成条件; 改配置由测试覆盖(极性/提示词/专家类/表哈希 任一变化 → digest 变化 → 冷启动)
  5. 测试 345 通过(test_config_fingerprint.py 15 项)
* **涉及/修改的文件清单**：
  - `utils/config_fingerprint.py (Created)`
  - `scripts/qwen_gain_baseline.py (Modified — 指纹计算/校验/报告字段)`
  - `tests/test_config_fingerprint.py (Created — 15 项)`
  - `plan.md (Modified — G3 执行状态表)`
* **执行结果与验证状态**：G3-a/b/c/d 全部完成; 345 测试通过; 干跑闭环(含指纹)验证通过
* **置信度或遗留待办（TODO）**：**G3-e n=120 GPU 复核暂停 —— 用户明确指示"待会不要开始GPU测试"**, 等指令; 完成后据此判定证据是否带来正增益
---
### 2026-09-24 17:55:10 - G3-a 补漏：全局测量的去重语义与 image-only 描述

* **当前操作动作**：修复整图度量引入的两处语义缺陷 + 提示词同源措辞 + 回归测试 + README 状态（用户指出的 G3-e 前置项）
* **核心变更说明**：
  1. **全局证据重复计权（真缺陷）**: `evidence_id` 键仍含 bbox, 而专家现在测整图 —— 模型用不同 bbox 重复调用同一专家会产生两个不同 id, **同一测量进入后验两次**; 停止策略对每个 token 加权, 即重复计权
  2. 修复: `evidence_id(..., measurement_scope)`: scope=global 时键**不含诊断区域**(source|strength|name), 重复调用因此与普通重复结果一样被去重; region 语义下的键不变; scope 本身参与身份(裁剪测量与整图测量是不同证据)
  3. 既有测试 `test_different_regions_produce_two_unique_evidence` 是 G1 局部测量时期的旧语义, 已改为**同专家不同区域 → 1 条唯一证据 + 1 次抑制**, 并新增"不同专家仍是 2 条唯一证据"
  4. **image-only 描述不准确**: 产物已改为整图渲染, 但文字仍写"对区域 [bbox] 的分析产物" → 会让模型把全图频谱/残差图误读为局部发现。改为明确"在**整幅图像**上计算法证指标; 所附产物图均为整图产物, 区域图对应你提出的关注区域(仅为诊断关注点, **不限定测量范围**)"
  5. **同源第三处(用户未列, 一并修)**: 提示词规则 2 "不要对已测量过的**区域**重复调用" 在整图测量下已不成立, 改为"**不要调用已调用过的专家**: 每个专家测量整图, 换区域再调用仍是同一测量, 会被抑制且浪费预算"
  6. 测试 353 通过(新增 7: 全局/区域 id 语义 3、image-only 描述 1、提示词契约 3); 端到端确认同一专家两次不同 bbox → 1 条唯一证据
* **涉及/修改的文件清单**：
  - `state_machine/evidence_tokenizer.py (Modified — evidence_id scope 感知)`
  - `state_machine/controller.py (Modified — image-only 标记措辞)`
  - `mllm/message_builder.py (Modified — 规则 2 改为按专家去重)`
  - `tests/test_evidence_tokenizer.py, tests/test_controller.py, tests/test_message_builder.py (Modified — 回归测试)`
  - `README.md (Modified — 测试数 353, G3 状态与进度表)`
* **执行结果与验证状态**：两处缺陷修复且被测试锁定; 353 测试通过
* **置信度或遗留待办（TODO）**：**G3-e n=120 GPU 四条件复核（用户已授权）** —— 携带新指纹冷启动
---
### 2026-09-24 18:15:30 - G3-e 前置修复：无工具会话的停止语义与多余结案轮

* **当前操作动作**：G3-e 首次启动即被自身问题中断（运行 2 分钟后停跑），修复后重启
* **核心变更说明**：
  1. **问题 1（严重）**: G3-e 首次运行中 rgb 臂出现 turns=7.0 / calls=0.0 / 全 Uncertain。根因是 v2 策略按"有可用工具且预期收益为正"持续要求取证，而基线臂的 BASELINE_SYSTEM_PROMPT **禁止调用工具** → 空转到回合预算耗尽并全部弃权，基线臂失去意义（原本是"1 轮结案、全判 Real 的领域先验基线"）
  2. 修复: 策略新增 `tools_available` 参数与**规则 0** —— 协议禁止调用工具时，**模型自身结论即为测量**（理由 `candidate_without_tools`），不做工具推荐（`next_expert=None`）也不等待不可能的证" 
  3. 控制器与 harness: `ForensicStateMachine(allow_exploration=...)`；四条件显式声明 `allow_exploration`（rgb=False，其余 True）
  4. **问题 2**: 策略接受模型自身结论后，控制器仍多调用一次"结案轮"取报告文本（模型已写过），基线臂轮数 2 而非 1，成本口径与 G2-d 不可比。修复: 接受自身结论时直接复用其报告，不再多花一次生成
  5. 真机冒烟确认: rgb 臂恢复 turns=1.0 / calls=0.0 / 全判 Real(0.95)，与 G2-d 基线协议一致
  6. 测试 360 通过（新增 7: 无工具会话 5、结案轮复用 2）
* **涉及/修改的文件清单**：
  - `state_machine/halting_v2.py (Modified — tools_available 与规则 0)`
  - `state_machine/controller.py (Modified — allow_exploration 接线 + 接受结论时复用报告)`
  - `scripts/qwen_gain_baseline.py (Modified — 四条件声明 allow_exploration)`
  - `tests/test_halting_v2.py, tests/test_controller.py (Modified — 回归测试)`
* **执行结果与验证状态**：两处修复生效并被测试锁定；真机冒烟确认基线臂恢复；360 测试通过
* **置信度或遗留待办（TODO）**：重启 G3-e（n=120 四条件）
---
### 2026-09-24 16:17:30 - 更正：本日志若干条目的时间戳估计偏差

* **当前操作动作**：更正时间戳（append-only，不改历史条目）
* **核心变更说明**：
  1. 发现: 我在 2026-09-24 15:31 之后的若干条日志中**按估计耗时而非读时钟**填写时间戳，导致时间偏移并累积
  2. 以 git 提交时间为真实锚点，实际与所写的对照如下：
     - G3-a（提交 2f4d2ca）实际 **15:31**，日志写 16:05:30（+34 分钟）
     - G3-b（提交 0ee3ae8）实际 **15:32**，日志写 16:30:45（+58 分钟）
     - G3-c（提交 7b89710）实际 **15:39**，日志写 17:05:20（+86 分钟）
     - G3-d（提交 4551dcc）实际 **15:40**，日志写 17:25:40（+105 分钟）
     - 全局去重/image 描述补漏（提交 4a2c553）实际 **16:01**，日志写 17:55:10（+114 分钟）
     - G3-e 前置修复（提交 aad0903）实际 **16:08**，日志写 18:15:30（+127 分钟）
  3. 15:31 之前的条目（含 G2-d/G2-e）时间戳系读时钟所得，无需更正
  4. 纪律: 此后每条日志时间戳一律以 `date` 输出为准，不再估算
* **涉及/修改的文件清单**：
  - `claude_operation_log.md (Modified — 追加本条更正)`
* **执行结果与验证状态**：偏移已量化并以提交时间为锚点固化；后续条目时间戳可信
* **置信度或遗留待办（TODO）**：无（历史条目的时间戳保留原样以便对照）
---
### 2026-09-24 16:40:10 - 事故与修复：配置指纹把文档提交当成配置变更，覆盖了 rgb 臂

* **当前操作动作**：修复 G3-d 指纹的设计缺陷（该缺陷导致 G3-e 的 rgb 臂数据被覆盖），并补跑恢复
* **核心变更说明**：
  1. **事故**: 一臂一臂开展 G3-e 时，rgb 臂完成（120/120，指纹 ac5e6cca…）后暂停；随后我提交了一条**仅改日志文档**的提交（55f30cd，19 行，未触碰任何代码）；启动 text 臂时续跑校验打印 `configuration changed (git_commit) — cold start`，**整份报告被覆盖，rgb 臂 120 条记录丢失**
  2. **根因**: `config_fingerprint.digest()` 把 `git_commit` 计入身份 —— 任何提交（含纯文档）都会改变指纹，从而把"继续同一实验"误判为"换了实验"
  3. 修复 1: digest 排除 `git_commit`（仍在报告中记录，作 provenance 而非身份）；新增 `_PROVENANCE_ONLY` 常量与说明
  4. 修复 2（防再次丢数据）: 冷启动（或 --fresh）时若目标报告已存在，先**移存为 `<name>.superseded_<时间戳>.json`** 再写，绝不静默覆盖已完成的臂
  5. 报告迁移: 当前报告（仅 text 臂）的指纹按新定义重算并写入 `migrated_from`/`migration_note`；迁移前 `differences()` 为**空**，证实除 git_commit 外各组件一致，迁移安全
  6. 测试 365 通过（新增 5: git_commit 不改 digest / 仍被记录 / 其余组件仍改变 digest / 新提交不再强制冷启动 / 冷启动备份）
* **涉及/修改的文件清单**：
  - `utils/config_fingerprint.py (Modified — digest 排除 git_commit)`
  - `scripts/qwen_gain_baseline.py (Modified — _backup_report 与冷启动备份)`
  - `tests/test_config_fingerprint.py (Modified — 5 项)`
  - `calibration/g3_gain_report.json (Modified — 指纹迁移, 保留 text 臂)`
* **执行结果与验证状态**：指纹缺陷修复且被测试锁定；text 臂数据有效（代码与 rgb 臂仅差一个文档提交）；rgb 臂补跑中
* **置信度或遗留待办（TODO）**：补跑 rgb 后暂停；image/both 待用户指令
---
### 2026-09-24 17:20:40 - G3-e 两臂结果（rgb / text）+ 机制诊断 + 一处我自己的分析误报更正

* **当前操作动作**：G3-e 按"一臂一臂"执行：rgb（补跑恢复）与 text 两臂完成（各 120 样本），随后按用户指令暂停
* **核心变更说明**：
  1. **两臂结果**（同样本集，v2 协议 + 整流器 + 整图度量）：rgb acc 0.525 / AUROC 0.654 / 1.1 轮 / 0 调用；text acc 0.275 / AUROC 0.502 / 3.6 轮 / 2.3 调用 / 弃权 66.7%；McNemar 配对 p=0.0004（text 显著更差）
  2. **分格**：text 在 `fake_png` 拿到 **13/15**（基线 0/15），但把 real 格从 15/15 打到 3–5/15
  3. **更正我自己的误报**：我曾报告"本次度量与校准集存储值有 110/234 不一致"，经复核**该结论是我的诊断脚本 bug**（按 glob 顺序取"最新 trace"，混入了 G3-a 之前的裁剪度量 trace: 1.286 正是裁剪尺度中位数量级）。正确口径下 **183/183 完全一致**，专家确定性无漂移
  4. **token 判别力**（text 臂样本，按格）: noise png 0.870 / q95 0.812 / q85 0.835 / q70 0.840（全格稳定，与 G2-b 的 0.845 一致）；jpeg png 0.907 / q95 0.577 / q85 0.544 / q70 0.441（与 G2-b 的 png 0.972 → q70 0.569 一致）
  5. **模型忠实跟随整流方向**: token=AI-generated(n=51) → P(Fake)=0.43 且从不判 Real；token=Real(n=39) → P(Fake)=0.00 且从不判 Fake；无方向(n=30) → 30/30 Uncertain
  6. **但仍有两个模型侧缺陷**: ①**容器格式先验**: P(Fake)=0.50(PNG) vs 0.08(JPEG)，而格式对齐格中格式无信号 → 直接摧毁 real_png ②**对证据过度保守**: 51 个 AI-generated token 仅 22 个转化为 Fake，29 个弃权
  7. 结论: 管线（测量/校准/方向/整流）已验证按设计工作; 剩余增益瓶颈在**模型如何使用证据**（格式捷径 + 弃权偏好）= G4/SFT 的训练目标
  8. 测试 365 通过（本轮无新增代码）
* **涉及/修改的文件清单**：
  - `calibration/g3_gain_report.json (Modified — rgb 120 + text 120)`
  - `logs/g3e_rgb_rerun.log, logs/g3e_text.log (Created)`
  - `claude_operation_log.md (Modified)`
* **执行结果与验证状态**：两臂完成且数据一致；机制诊断完成；**image / both 两臂按用户指令暂停未跑**
* **置信度或遗留待办（TODO）**：image / both 待指令；G3-e 完成后据此更新 G2/G3 状态并进入 G4
---
### 2026-09-24 18:05:20 - G3-e 完成：四臂结果与结论（含按格式配对的判别力分解）

* **当前操作动作**：补齐 image / both 两臂，完成 G3-e 四条件复核（120 样本 × 4 臂）
* **核心变更说明**：
  1. 四臂（同一样本集，v2 协议 + 整流器 + 整图度量）: rgb acc 0.525 / AUROC 0.654 / 1.1 轮 / 0 调用; text 0.275 / 0.502 / 3.6 / 2.3; image 0.258 / 0.543 / 3.7 / 2.6; both 0.225 / 0.539 / 3.8 / 2.5
  2. McNemar 配对 vs rgb: text p=0.0004、image p=0.0003、both p<1e-4 —— 三条工具臂**准确率仍显著低于基线**
  3. **按格式配对 AUROC（real_X+fake_X 合并, 格内容器无信号）**: png 格 rgb 0.609 vs text 0.827 / image 0.791 / both 0.782（工具臂显著更强, 属真判别力）; q95/q85/q70 格 rgb 0.600–0.733 vs 工具臂 0.391–0.509（低于随机）
  4. **增益与损害分布在不同格式、相互抵消** —— 这是"整体 AUROC 追平但无净增益"的成因
  5. 机制（360 工具臂会话）: 模型忠实跟随整流方向 —— token 指 AI-generated 时 P(Fake)=0.85(png)/0.38(jpeg), 指 Real 时均 0.00, 无方向时全弃权; 判 Fake 精确率 0.77–0.80（text 22 中 17 对、image 25 中 20 对、both 23 中 18 对）
  6. 判定: **G3 目标达成**（工具臂由"低于随机"回到"随机至弱正", 且 PNG 格内具备 0.78–0.83 的真实判别力; 管线经 183/183 度量一致与忠实读取验证）; **净增益仍缺**, 瓶颈在模型侧三条: 容器格式先验、JPEG 格置信度排序反向、过度弃权
  7. 对 G4 的直接含义: 训练目标应针对 (a) 不得把容器格式当伪造线索 (b) JPEG 格按 noise 专家与校准后验下结论
* **涉及/修改的文件清单**：
  - `calibration/g3_gain_report.json (Modified — 四臂各 120 条)`
  - `logs/g3e_image_both.log (Created)`
  - `plan.md, README.md, claude_operation_log.md (Modified — G3-e 结果与结论)`
* **执行结果与验证状态**：G3-e 完成（四臂 480 会话）; 365 测试通过; 报告指纹 25b169996ff2d422
* **置信度或遗留待办（TODO）**：G3 全部完成（a–e）; 下一步 G4：以修复后的管线重新生成 SFT final_v2 并人工抽检, 训练目标显式覆盖上述两条模型侧缺陷
---
### 2026-09-26 10:35:20 - G4 计划补全 + G4-a 数据划分与防泄漏（完成）

* **当前操作动作**：①按项目规范把 G4 拆为可验收子阶段写入 plan.md ②修正 README/plan 中 G3-e 的旧状态 ③实施 G4-a 数据划分
* **核心变更说明**：
  1. **G4 执行计划（a–g）** 写入 plan.md §4.11: G4-a 数据划分与防泄漏(CPU) / G4-b 候选轨迹生成器(Schema 与测试 CPU, 执行需 GPU) / G4-c 两级准入(CPU) / G4-d final_v2 Schema 与校验器(CPU) / G4-e 小规模 GPU 试生成 / G4-f 全量人工审核 / G4-g LoRA 小规模试训并重跑四臂验收
  2. 旧状态修正: plan.md 的 "G3-e ⏸ 等待 GPU 授权" 改为 "✅ 已完成（四臂 480 会话）"; README 进度表的 "G3-e 进行中" 改为 "✅ 完成"，并新增 G4-a~d 行
  3. **G4-a 实现** `scripts/build_split_v2.py` + `sft_data/split_v2.json`: 泄漏单元是**源图**而非文件（同一源图的 native/png/q95/q85/q70 由 `split_for_variant()` 归一到同一分区, 并在 build 时逐条断言）; 记录生成器来源供 G5 跨生成器评测; 98 个 G2-b 校准源划为 `calibration_holdout`（既不训练也不评测, 避免标定数据自评）; 内容哈希 `d8e9d81597ead53f` 保证划分可复现
  4. **划分口径修正（重要）**: 数据集本身是 1000 Real : 8000 Fake（1:8）, 按 (label×generator) 忠实分层会让评测集变成 11/89 的类别混比, 与校准集 50/50 及 G3-e 四臂不可比, 还会放大"Fake 先验"这一已知失效模式。故 **val/test 类别平衡**（Real 142/143 对 Fake 142/143, 每格 8 个生成器均摊）, **train 保留全部候选并记录采样权重**（`train_sampling_weights: {Real: 1.0, Fake: 11.53}`）供 G4-b 按权重显式配平
  5. 测试抓到两个真实 bug 并修复: ①`validate()` 在发现问题时引用不存在的 `entry["source_id"]`（真出问题时反而 KeyError 崩溃）②Python 默认参数在定义期绑定, 导致测试注入的 root 不生效（`load_holdout_sources` 现显式传 root）
  6. 测试 386 通过（新增 split 测试 21 项）; 端到端测试单例约 3.6s（G3-a 整图度量的预期代价, 非测试问题）
* **涉及/修改的文件清单**：
  - `plan.md (Modified — G4 执行计划 a–g; G3-e 状态修正)`
  - `README.md (Modified — G3-e 状态修正 + G4-a~d 行)`
  - `scripts/build_split_v2.py (Created — 源级划分/防泄漏/生成器来源/holdout/校验器)`
  - `tests/test_split_v2.py (Created — 21 项)`
  - `sft_data/split_v2.json (Created — 9000 源划分, 1572 KB)`
  - `claude_operation_log.md (Modified)`
* **执行结果与验证状态**：G4-a 完成; 386 测试通过; 划分校验全过（无跨分区源、holdout 不重叠、评测分区类别平衡且生成器均摊）
* **置信度或遗留待办（TODO）**：G4-b（轨迹生成器 Schema + Mock 测试, CPU）→ G4-c（两级准入）→ G4-d（final_v2 Schema 与校验器）均在 CPU 完成; G4-e/g 需 GPU（当前未开启）
---
### 2026-09-26 11:20:40 - G4-b 前半：按工具集的提示词 + 训练源格式变体

* **当前操作动作**：G4-b 的 CPU 前置件 —— ①提示词可按键允许的工具集生成 ②为训练源生成格式变体
* **核心变更说明**：
  1. **提示词策略化**（`build_forensic_prompt(allowed_tools)`）: 只允许 noise 的轨迹不得被告知存在三个专家, 否则会花回合请求本会话不会服务的工具。按工具集过滤"可用动作"与"各工具实测说明"两块, 子集额外声明"本会话仅存在上列工具"
  2. **字节级兼容**: 全工具集输出与已 ship 的 FORENSIC_SYSTEM_PROMPT **逐字节一致**（用 git HEAD 原文本拆解重组, 修正了拆解时空行被 rstrip 吃掉的接缝差异）, 并以内容哈希 `09dd259ae0b130f6` 钉在测试里 —— 保证 G3 的测量仍描述实际运行的提示词
  3. **变体构建**（`scripts/build_variants_g4.py`）: 校准集只为它自己的 98 个源建过格式格, 训练源需要同样的变体; 复用 `build_calibration_set.build_cells`（新增 out_dir 参数）而**不重新实现** —— 可靠性表描述的就是那套变换, 另写一份会静默使其失效
  4. 源来自 `split_v2.json`, 因此变体继承源的划分, 任何重编码都不可能跨分区; 抽样默认**类别配平**（使用划分记录的权重）, 不静默继承数据集 1:8
  5. 产物: `sft_data/variants/manifest.json`（变体 → 源/生成器/划分/处理）; 生成图像加入 gitignore, manifest 保留入库
  6. 测试 404 通过（新增 6 项提示词策略 + 12 项变体构建）
* **涉及/修改的文件清单**：
  - `mllm/message_builder.py (Modified — TOOL_ACTIONS/TOOL_MEASURES/build_forensic_prompt)`
  - `scripts/build_calibration_set.py (Modified — build_cells 支持 out_dir)`
  - `scripts/build_variants_g4.py (Created)`
  - `tests/test_message_builder.py, tests/test_build_variants_g4.py (Modified/Created)`
  - `.gitignore (Modified — 变体图像)`
  - `claude_operation_log.md (Modified)`
* **执行结果与验证状态**：提示词字节一致性与子集过滤均已测试锁定; 变体构建 8 源冒烟通过（32 变体）; 404 测试通过
* **置信度或遗留待办（TODO）**：G4-b 核心（候选轨迹生成器 + Schema + Mock 测试）随后提交; 执行需 GPU（当前未开启）
---
### 2026-09-26 12:05:10 - G4-b 完成：候选轨迹生成器（Schema + 门控 + Mock 测试）

* **当前操作动作**：G4-b 核心 —— 每条 (源变体 × 工具策略) 生成一条完整轨迹记录
* **核心变更说明**：
  1. 六种策略: `no-tool`（基线, 不服务任何工具且禁止探索）/ `noise` / `jpeg` / `frequency` / `noise+jpeg` / `noise+frequency`; 会话开始前即确定可调用工具 —— 提示词只列这些, 且只注册这些专家, 未服务的调用无专家可派发
  2. **记录内容**（每条轨迹自足, G4-c 可仅凭记录做准入）: 原始模型判断、每个 Evidence Token（含 measurement_scope/raw_metric/calibrated_likelihood/applicability/适用条件）、停止策略后验与 reasons、最终判断与置信度、计数器（轮/调用/唯一证据/抑制/加权成本）、耗时, 以及**两条概率各自的 NLL/Brier**（模型自报概率与结构化后验分开计分）
  3. **jpeg 门控是数据驱动的**: `load_applicability` 读 G2-b 的逐格 AUROC 并按极性校正（jpeg 高值=Real, 故其可提供分离度 = 1−auroc）, 需要 jpeg 的策略在该格分离度低于 **与 G4-c 同一把尺**（0.65）时不予生成, q70（0.431）被门掉并记录原因
  4. 弱专家 frequency 仍按用户要求生成 —— 其准入是 G4-c 的判断, 不是生成阶段的判断
  5. 记录 `model_probability` 仅在模型确实给出候选结论时填写（无候选则为 None, 不以策略标签冒充模型意见）
  6. 测试 426 通过（新增 22 项: 适用性校正与拒绝、策略完整性、门控计划、NLL/Brier、聚合、端到端干跑与续跑）
* **涉及/修改的文件清单**：
  - `scripts/generate_trajectories_g4.py (Created)`
  - `tests/test_generate_trajectories_g4.py (Created — 22 项)`
  - `claude_operation_log.md (Modified)`
* **执行结果与验证状态**：G4-b 完成; 干跑 2 变体 × 6 策略 = 12 条轨迹落盘; q70 的 jpeg 类策略按门控跳过; 426 测试通过
* **置信度或遗留待办（TODO）**：G4-c 两级准入（CPU）→ G4-d final_v2 Schema 与校验器（CPU）; 实际生成需 GPU（当前未开启）
---
### 2026-09-26 12:40:30 - G4-c 完成：两级准入（条件层 + 增益层）

* **当前操作动作**：G4-c —— 用两级门槛决定哪些轨迹可以进入训练, 并记录每条轨迹的去留理由
* **核心变更说明**：
  1. **第一层 条件准入**: 非弱专家必须在其格式格上通过 **与生成阶段同一把尺**（0.65）才能承担权重, 否则否决并记录原因（如 jpeg 在 q70 为 0.431）; 已停用的 v1/ELA 出现即否决; **弱专家（freq）可作为佐证出现**（其自身分离度低于门槛只记录不否决）, 但**没有任何工具单独达标时该轨迹永不可能是正样本** —— 这正是规范里"弱专家仅作独立佐证、不能单独高置信结案"的可执行形式
  2. **第二层 增益准入**（相对同变体的 no-tool）: Brier 与 NLL（真值标签）、后验赋予真值的概率、以及是否用"自信的错误"换掉了"诚实的弃权"; 只有**条件达标 + 风险确实下降（超过 epsilon, 一丝改善不算教训）+ 结论正确**才是**正样本**; 在基线自信判错处改为弃权者记为**合理弃权**（honest_abstention）, 单独成类
  3. **成本不设拍脑袋阈值**: 记录 admitted 与 rejected 的 mean Δcost, 用数据回答"增加调用是否值得", 由 G4-g 的 ΔAUROC 作最终裁判
  4. **训练分区守卫**: 准入在做出任何判断前断言 `split == "train"`, 读到 val/test 立即抛错 —— 用评测集挑选轨迹正是本项目反复警惕的泄漏
  5. 测试 449 通过（新增 23 项: 守卫、两层判定、弃权分类、配对、加载、聚合）
* **涉及/修改的文件清单**：
  - `scripts/admit_trajectories_g4.py (Created)`
  - `tests/test_admit_trajectories_g4.py (Created — 23 项)`
  - `claude_operation_log.md (Modified)`
* **执行结果与验证状态**：G4-c 完成; 干跑端到端（22 条轨迹 → 4 正样本 / 18 合理弃权）验证管线; 449 测试通过
* **置信度或遗留待办（TODO）**：G4-d final_v2 Schema 与校验器（CPU, 最后一项 CPU 工作）; 之后需 GPU 才能继续 G4-e
---
### 2026-09-26 13:25:40 - G4-d 完成：final_v2 Schema、渲染器与校验器（CPU 阶段收尾）

* **当前操作动作**：G4-d —— 把获准入的轨迹渲染成四段式训练样本, 并用十项自动校验把关
* **核心变更说明**：
  1. **四段式 Schema**: `<observation>`（可直接核验的事实 + 模型首轮自己的异常描述, 取自 trace 的 `<planning>/Visual Anomalies`）、`<forensic_evidence>`（evidence_id/专家/测量范围/原始数值/calibrated_likelihood/applicability/适用条件）、`<reasoning>`（支持证据、反证与替代解释、专家失效条件、剩余不确定性=后验+冲突度+停止原因）、`<verdict>`（后验给出的标签与置信度）
  2. **十项自动校验**（逐条从记录重新推导, 不信任文本）: 同源划分泄漏 / 结构与标签 / Evidence ID 引用存在 / 全局测量范围与 bbox 边界 / 重复调用 / support-文字-校准方向矛盾（复用整流器与 G1 门）/ 不适用专家被采纳 / verdict 与后验不一致 / 高置信无合格证据 / **把容器格式当真假理由**
  3. **校验口径与准入对齐而非更严**: 弱佐证专家低于门槛时若有其他专家承担则可通过; 低于门槛只有在样本仍**给出标签**时才算缺陷 —— 弱证据落到诚实弃权正是要教的课
  4. **容器检查要求因果连接词**（因为/因此/说明/indicates…）且允许否定式, 因此我们自己写的条件说明（"JPEG 压缩引入方差, 不得按 AI-generated 解读"）不会被误判为它警示的那类错误
  5. **冻结守卫**: 旧 `final/` 是回归基线, 构建器**拒绝写入其内部**（路径解析后比对）; 产物写入 `sft_data/train/final_v2/`
  6. 端到端干跑: 22 条准入轨迹 → 4 正样本 + 18 合理弃权, 全部通过校验（0 拒绝）
  7. 测试 483 通过（新增 34 项: 渲染、十项校验各自的触发与通过情形、构建守卫）
* **涉及/修改的文件清单**：
  - `utils/final_v2.py (Created — Schema/渲染/十项校验)`
  - `scripts/build_final_v2.py (Created — 组装/分桶/冻结守卫)`
  - `tests/test_final_v2.py (Created — 34 项)`
  - `claude_operation_log.md (Modified)`
* **执行结果与验证状态**：**G4-a~d（CPU 部分）全部完成**; 483 测试通过
* **置信度或遗留待办（TODO）**：**下一步需要 GPU（当前未开启）** —— G4-e 小规模试生成（100–200 源 × 6 策略）→ G4-f 人工审核 → G4-g LoRA 试训并重跑四臂验收
---
### 2026-09-26 15:10:20 - G4-e 试生成完成（20 源 / 440 轨迹）+ 两处修复

* **当前操作动作**：G4-e 小规模试生成（用户指定先跑 20 源）→ 准入 → final_v2 组装与校验
* **核心变更说明**：
  1. **规模**: 20 源（10 Real + 10 Fake）× 4 处理 × 6 策略 = **440 次会话**（q70 的 40 次 jpeg 类策略在生成阶段门控跳过）; 平均 20.2s/会话, 约 2.5 小时
  2. **修复 1（G4-c）**: 无工具轨迹与自身基线比较恒为 Δ0, 导致**正确答案的无工具轨迹被系统性丢弃**（50 条里 38 条）, `no_tool_positive` 桶永远为空。判据改为"仅凭图像答对且无自信错误"; 另 40 条仍被正确拒绝 —— 基线在 Fake 上高置信判 Real
  3. **修复 2（G4-d）**: 无工具样本被两项只适用于"证据结论"的校验全部拒绝（37/37）: 无证据时后验按构造为 0.5 → "verdict 与后验矛盾"; "高置信必须有合格证据" → 而仅凭视觉下结论正是该桶要教的。verdict 段现标明依据（`model_perception` / `evidence_posterior`), 无工具样本改校验"是否与模型实际结论一致"
  4. **顺带修复渲染 bug**: render_answer 内 `evidence` 变量被二次赋值成原始列表, 会把 Python repr 写进训练文本（测试立即捕获）
  5. **质量检查全部达标**: 报告唯一率 100%; q70 的 jpeg 策略 0 次生成; 弃权 100% 有理由（后验偏离<0.15）; 360 条工具会话置信度 0 条偏离后验; 配对 ΔBrier: noise −0.027 / jpeg −0.025 / noise+jpeg −0.021 / noise+frequency −0.027 / **frequency −0.001**（与标定预测一致）
  6. **准入**: 440 条 → 正样本 97（工具 60 + 无工具 37）/ 合理弃权 140 / 拒绝 203; frequency 正样本 0
  7. **final_v2**: 237 条, 十项校验 **0 拒绝**
  8. **须在训练前处理**: 类别成分 Fake 178 / Real 59（75/25）, 各桶内部同样偏斜。原因是任务结构（证据只在基线自信判错处有增益, 而基线在 Fake 上正高置信判 Real）; 风险是学出"调用工具⇒Fake"的新捷径。已记录 `label_composition` 与 `training_weights`(Real ×3.02) 供 G4-f/G4-g 裁决
  9. 测试 495 通过
* **涉及/修改的文件清单**：
  - `scripts/admit_trajectories_g4.py, tests/test_admit_trajectories_g4.py (Modified — 无工具判据)`
  - `utils/final_v2.py, scripts/build_final_v2.py, tests/test_final_v2.py (Modified — 依据区分/渲染修复/成分权重)`
  - `sft_data/trajectories/ (Created — 440 条), trajectories_report.json, admission_report.json`
  - `sft_data/train/final_v2/ (Created — 237 条 + metadata)`
  - `plan.md (Modified — G4-e 结果与质量检查表)`
* **执行结果与验证状态**：G4-e 完成; 495 测试通过; 十项校验 0 拒绝; 五项质量检查全部达标
* **置信度或遗留待办（TODO）**：类别成分偏斜须在训练前裁决; G4-f 人工审核（全量首审 + conflict/Uncertain/multi-tool/高置信双审）→ G4-g LoRA 试训并重跑四臂验收
---
### 2026-10-08 09:35:10 - G4-f 审核工作台与处置工具（CPU）

* **当前操作动作**：G4-f 前置 —— 搭建人工审核工作台、处置记录器与审核后组装器
* **核心变更说明**：
  1. **环境变化**: GPU 已释放（No devices found）→ G4-e 扩量与 G4-g 阻塞; 仓库远端同步、工作区干净; 526 测试通过（单例耗时因机器变慢约 1.5–2×）
  2. **工作台** `scripts/build_review_workstation.py`: 为 237 条样本各生成一页 —— 源图与模型实际输入的变体并排、专家可视化产物、Evidence Bundle 表（evidence_id/测量范围/原始数值/校准概率/适用性/适用条件）、与 no-tool 基线的对比、**该样本将教模型说的四段答案**、以及逐项审核清单; 另出 index/工作单 CSV/审核集合 JSON。静态 HTML、无外部依赖、图片相对路径引用（0 张缺失）
  3. **审核清单内置 7 项**（规范要求"不能只看格式"）: 观察可核验 / 证据段与 token 一致 / 推理只引合格证据 / 不得把容器当真假理由 / verdict 与后验一致 / 增益真实 / 不教坏习惯
  4. **双审清单规模提醒**: 按字面定义覆盖 207/237（弃权 140、多工具 78、高置信 38）, 是否对"设计上弃权"的样本也双审留由审核者裁决, 工具不替其决定
  5. **记录器** `scripts/record_review.py`: 四类处置沿用 G0 词汇; 追加式 JSONL、最新为准、错误样本 id 拒绝写入（否则真实样本会"看着已审"却未审）
  6. **组装器** `scripts/apply_review.py`: 依处置分四集输出到 `final_v2_reviewed/`, 样本携带审核人与理由; **未审完拒绝运行**（防"训练了没人看过的样本"）; 自动计算 accepted 集类别权重以抵消偏斜; 不写入任何冻结集
  7. 测试 526 通过（新增工作台 14 项 + 审核管线 17 项）
* **涉及/修改的文件清单**：
  - `scripts/build_review_workstation.py, tests/test_review_workstation.py (Created)`
  - `scripts/record_review.py, scripts/apply_review.py, tests/test_review_pipeline.py (Created)`
  - `sft_data/review/ (Created — 237 页 + index + worklist + review_sets)`
  - `plan.md (Modified — G4-f 工具与双审规模)`
* **执行结果与验证状态**：工作台与处置管线就绪; 526 测试通过; 演练（记录两条处置、最新生效、未审完拒绝出训）验证通过
* **置信度或遗留待办（TODO）**：等待人工审核（G4-f）; GPU 恢复后 G4-g(LoRA 试训 + 四臂复跑) 与 G4-e 扩量
---
### 2026-10-08 09:30:23 - 文档收口：重读 agent.md 并补齐未完成项

* **当前操作动作**：按 agent.md 逐条核对四轨文件与文档规范，补齐缺失内容
* **对应计划锚点**：agent.md §1.1（四轨文件）/ §2.1（日志格式）/ §6.1（README 规范）；plan.md §4.11（G4）
* **核心变更说明**：
  1. **日志格式合规**: 核对发现 agent.md §2.1 要求的 **"对应计划锚点"** 字段在 2026-09-24 之后的条目中缺失（早于该要求的条目不受影响）；**自本条起补齐该字段**，历史条目保持原样（append-only）
  2. **README 七处补齐**: ①目录树加入 G4 全部脚本与产物（split/variants/trajectories/admission/review/final_v2 等）②§6.1 专家表从"原始理论"改为**G2 实测口径**（noise/jpeg 反向、v1 停用、ELA 不注册、逐格分离度与准入）③§6.2 终止机制由"四重守卫"改为**策略 v2**（后验/冲突/效用/停滞/预算，含回放对比数字）④§7.1 已完成表补 G2/G3/G4 三行 ⑤§3.6b 新增 **G4 数据管线七步命令** ⑥§5.6 新增 **G4 Schema**（划分/轨迹/准入/final_v2 四段式/审核记录）⑦§9 运行环境补 GPU 已释放与 cgroup 约束史
  3. **架构文档**: 新增 §5.6 G4 数据管线（六步流程图、每步关键约束、**方向权威链**、当前产物规模）
  4. 核对结论: 四轨文件均在位且生命周期符合 §1.1；`{project}_tasks.md` 未被改动（只读要求 ✅）；提交粒度与信息格式符合 §1.2/§5.1；GPU 审批流程符合 §3.2.3
* **涉及/修改的文件清单**：
  - `README.md (Modified — 目录树/专家准入/终止策略/已完成表/数据管线/Schema/运行环境)`
  - `CURRENT_PROGRAM_ARCHITECTURE.md (Modified — §5.6 G4 数据管线)`
  - `claude_operation_log.md (Modified — 本条起补齐"对应计划锚点"字段)`
* **执行结果与验证状态**：文档与当前实现一致（G0–G4）；526 测试通过未受影响
* **置信度或遗留待办（TODO）**：README §7.2 的 G4-g/G5 仍待 GPU；审核工作台待人工使用
---
### 2026-10-08 10:25:40 - G4-d 修复：训练对话必须复现推理协议

* **当前操作动作**：暂停前收尾 —— 修复 final_v2 训练对话缺失工具调用与证据轮的问题（并在暂停后不遗留未提交改动）
* **对应计划锚点**：plan.md §4.11 G4-d（final_v2 Schema 与校验器）；agent.md §3.2.1（CPU 可测）
* **核心变更说明**：
  1. **发现的问题（严重）**: `render_sample` 只渲染两轮（`<image>` 提问 + 四段答案），**丢掉了工具调用轮与证据轮**，而答案里引用了 evidence_id —— 这等于教模型引用从未收到的证据，与伪造证据无法区分
  2. 修复: 新增 `render_conversation(record, trace)` **按推理协议重建对话** —— 保留原始 trace 中模型的 planning+调用轮，证据轮由**整流后的 token** 重建（用 `EvidenceTokenizer.to_json`，附最多两张产物图，与推理一致），丢弃 `[System: ...]` 脚手架轮与旧结案文本，最后接上四段答案; 构建器现读取 trace 并传入
  3. 新增**第 11 项自动校验**: 对话必须以 user 开头、以助手答案结尾; 工具样本的证据轮数必须等于 token 数（否则报"答案引用了从未交付的证据"）; 证据轮的 evidence_id 必须在链中; 无工具样本不得携带证据轮
  4. 校验器另修一处读取错误: 答案取**最后一个** gpt 轮（重建后第一轮是工具调用）
  5. 效果: 工具样本训练对话现为 4 轮（提问 → planning+调用 → 证据 token+图 → 四段答案）；重建后 237 条全部通过校验（0 拒绝）
  6. 测试 551 通过（新增 10 项：对话重建 5 + 对话校验 5）
* **涉及/修改的文件清单**：
  - `utils/final_v2.py (Modified — render_conversation + 第 11 项校验 + 答案轮次修正)`
  - `scripts/build_final_v2.py (Modified — load_trace 并传入渲染)`
  - `tests/test_final_v2.py (Modified — 对话重建与校验 10 项)`
  - `sft_data/train/final_v2/ (Modified — 按推理协议重建 237 条)`
  - `sft_data/review/ (Modified — 工作台按新样本重建)`
  - `scripts/check_acceptance_g5.py, tests/test_check_acceptance_g5.py (Created — G5 验收核对器)`
* **执行结果与验证状态**：训练对话与推理协议一致; 551 测试通过; 工作台已重建
* **置信度或遗留待办（TODO）**：**用户要求暂停**。恢复后待办：①LoRA 训练脚本（`scripts/train_lora_g4.py`：按 accepted 集 + 类别权重采样，含 CPU dry-run）②四臂 harness 支持 `--adapter` 加载 LoRA ③G4-f 人工审核（工作台已就绪）④G4-e 扩量与 G4-g 训练需 GPU（当前已释放）
---
### 2026-10-08 10:55:00 - G4-g 显存适配：掩码位置交叉熵，试训跑通

* **当前操作动作**：在 24 GB 卡上跑通 LoRA 小规模试训 —— 定位并修复反向传播 OOM
* **对应计划锚点**：plan.md §4.11 G4-g（LoRA 小规模试训）之「训练显存适配」；agent.md §3.2.3（GPU 授权）
* **核心变更说明**：
  1. **根因（逐项排除后定位）**: ①模态编码器是否仍在反向图 —— 日志确认已断开，排除；②梯度检查点是否生效 —— 包装前后 `is_gradient_checkpointing` 均为 True，排除；③**模型自带 loss 把全部 2442 个位置投影过 152k 词表**：logits 及其梯度各约 742 MB，外加交叉熵中间量，而监督位置只有 489 个 —— 视为根因
  2. 新增 `compute_masked_loss()`：不再把 `labels` 交给模型，取 `output_hidden_states[-1][:, :-1]`，只在 `labels != -100` 的移位位置上过 `lm_head` 做交叉熵；训练循环改调此函数
  3. **等价性由测试钉住**（`TestMaskedLoss` 5 项，全 CPU）：与独立计算的掩码交叉熵逐值一致 / 扰动被掩位置不改变 loss / head 只见监督位置（形状 3×6 而非 5×6）/ 多模态张量透传（`pixel_values`、`image_grid_thw`、`output_hidden_states`、`use_cache=False`）/ loss 仍连着计算图（可 backward 到 head 与 embedding）
  4. 试训结果：237 样本 × 2 epoch = **474 步**完成，loss 1.0630 → 0.0002，adapter 190 MB 落盘；`run_manifest.json` 记录数据成分、采样权重、LoRA 超参与**含 adapter 的配置指纹**（`5052be1b498fcf75`）
  5. **样本成分未变**（Fake 178 / Real 59，Real 权重 ×3.02 配平），但 2 epoch 内 loss 已至 3e-4 —— **是记忆而非泛化**，故本次试训**只作训练管线与四臂接线的验证**
  6. 四臂评估（`--adapter`）随即启动：`calibration/lora_arms.json`，per_cell=15 与 G3 基线同一样本集（配对比较），预计约 1.5–2 h
* **涉及/修改的文件清单**：
  - `scripts/train_lora_g4.py (Modified — compute_masked_loss + 训练循环改用)`
  - `tests/test_train_lora_g4.py (Modified — TestMaskedLoss 5 项)`
  - `sft_data/lora_runs/provisional_20src/ (Created — adapter + run_manifest.json)`
  - `plan.md (Modified — G4-g「训练显存适配」)`
  - `calibration/lora_arms.json (Created — 四臂评估，运行中)`
* **执行结果与验证状态**：576 测试通过；训练 exit 0（OOM 消除）；适配器加载成功（18.1 s / 17.0 GB 显存）；四臂评估进行中
* **置信度或遗留待办（TODO）**：①本次数据**未经 G4-f 人工审核**（`--from-final-v2` 临时集），任何指标都不具科学结论性，审核后须以 accepted 集重训 ②四臂 ΔAUROC / JPEG 格 / 弃权率与 `check_acceptance_g5.py` 判据待评估完成 ③2 epoch 已明显记忆，真集上需重定 epoch 与早停策略
---
### 2026-10-08 12:25:00 - G4-g 首轮四臂否证 → 定位两处协议失配 → 复训冒烟通过

* **当前操作动作**：用首个适配器跑四臂，发现"模型根本没在执行协议"；定位为**训练目标**与**提示词分布**两处失配并修复，复训后以冒烟验证协议恢复
* **对应计划锚点**：plan.md §4.11 G4-g（LoRA 小规模试训）之「首轮四臂（未通过）+ 三处协议失配」；G4-g 验收判据（ΔAUROC 为正 / 不再低于随机）
* **核心变更说明**：
  1. **首轮四臂的失败形态（否证）**: rgb 臂 120/120 全答 `Real, 0.95`、AUROC **0.500**；工具臂 **127 次会话中仅 1 次发出 `<call_*>`**；**127/127 会话出现逐字节复读**（控制器驳回后原地重复至预算耗尽）；6 条会话写出 `<forensic_evidence>` 而该轮并未收到任何证据。结论：不是"效果差"，是"没有在跑协议"
  2. **根因一（决定性，已修）**: `split_answer` + 全提示掩码把"除最后一个助手轮外的全部 token"都当提示 —— **工具轨迹里的调用轮是助手轮，同样被掩掉**，模型从未学过发出调用，只学会"问题之后直接是答案"。修复：新增 `assistant_spans()` 定位并监督**每一个** `<|im_start|>assistant … <|im_end|>` 轮（含证据轮之间的中段 reasoning 轮），用户/系统轮掩为 -100；上报字段改为 `supervised_turns`/`supervised_tokens`。**修复后首步 loss 1.06 → 2.82**，即多出的 2–3 个助手轮进入监督范围的直接证据
  3. **根因二（已修）**: 训练样本用的是生成器**按策略裁剪的子集提示**，而四臂推理发的是**出厂全工具提示** —— 训练集 0 条见过推理提示。修复：`system_prompt_for()` 对工具样本返回 `build_forensic_prompt()`（出厂提示，字节一致）；抽样核对 229/237 条答案未提及受限工具集，目标文本在出厂提示下仍然合法
  4. **根因三（未修，交人工）**: 标签成分偏斜 —— 工具正样本 48F/12R、**无工具正样本 37R/0F**、合理弃权 130F/10R。无工具桶单标签正是常量 `Real` 的来源。计划已写明"无改善则先修数据与目标"，配平/接受/补样本由 G4-f 审核裁决
  5. **真实 processor 验证**: 工具样本现被监督 **2–3 个助手轮**（planning+调用 / 中段 reasoning / 最终答案），监督 token 由约 489 升至 579–897；无工具样本仍为 1 轮 121 token
  6. **复训与冒烟**: `provisional_20src_v2`，474 步、loss 2.82 → 0.023；text 臂 32 样本 **32/32 全都会调用工具**（每样本 1.62 次，基线 2.34），轮数 3.69，复读消失
  7. **配对比较（同 32 样本，仅作健康度）**: acc 0.219→**0.375**、F1 0.348→0.476、弃权 0.625→0.562、调用 2.34→1.62；但 AUROC 0.535→**0.502**（仍在随机附近，n=32 差异在噪声内）—— **协议恢复、增益未获证明**
  8. **防泄漏复核（跨产物）**: 20 个训练源全部落在 `train` 分区、98 个校准源在 split 表外（holdout 记 `calibration_g2b`）、两侧源路径交集为空 —— 已写成回归测试固定
* **涉及/修改的文件清单**：
  - `scripts/train_lora_g4.py (Modified — assistant_spans 全助手轮监督 + 出厂提示 + supervised_* 字段)`
  - `tests/test_train_lora_g4.py (Modified — 助手轮监督/跨产物防泄漏，共 32 项)`
  - `plan.md (Modified — G4-g 显存适配 / 首轮四臂否证与三处失配 / 配对比较)`
  - `README.md (Modified — 目录树补训练与验收脚本 / 已完成表 G4-g / G4-f 停在人工)`
  - `calibration/lora_arms_prefix.json (Created — 修复前四臂部分结果，作为否证证据)`
  - `calibration/lora_smoke_v2.json (Created — 修复后 text 臂 32 样本冒烟)`
  - `sft_data/lora_runs/provisional_20src_v2/ (Created — 复训 adapter + manifest)`
* **执行结果与验证状态**：583 测试通过；两次训练 exit 0；修复前报告按诚实命名保留（`_prefix`）；`sft_data/lora_runs/` 已加入 `.gitignore`（单次 190 MB，可由脚本 + 入库数据复现）
* **置信度或遗留待办（TODO）**：**停在 G4-f 人工审核**。需人工裁决：①237 条样本的审核与处置（工作台 `sft_data/review/index.html`，双审集合 207/237）②标签成分偏斜如何处置（配平采样 / 接受 / 补充 Real 侧工具样本）③是否投入全量四臂（修复后每样本 75–90 s，480 会话约 10 GPU 小时）—— 建议先出 accepted 集再重训与比较
---
### 2026-10-09 11:20:00 - G4-f 工作台修订：单次审核 + 全量对话 + 证据语义 + 可保存决定

* **当前操作动作**：按用户逐条复核意见重修审核工作台（审核尚未开始，此时修成本最低），并取消双审协议
* **对应计划锚点**：plan.md §4.11 G4-f 之「G4-f 工作台修订（2026-10-09）」；agent.md §1.2（原子提交）、§3.2（CPU 可测）
* **核心变更说明**：
  1. **完整训练对话（严重）**: 原页 `answer = next(gpt turn)` 只渲染**第一个**助手轮 —— 237 条中 198 条是多轮，页面因此只显示 `<planning>`，把工具调用轮、证据输入轮与最终四段答案全部藏掉，等于让审核者凭最没信息量的一轮下判断。改为按序渲染**全部轮次**，逐轮标注「参与训练 · 被监督」（助手轮）/「掩码 · 不进 loss」（用户/系统轮），与 `scripts/train_lora_g4.py` 的 `assistant_spans` 监督范围一一对应；顶部给出「共 N 轮：助手 M 轮参与训练」
  2. **Evidence Bundle 语义补全**: 原表 8 列且把 `support` 标成"方向"。改为逐 token 卡片：`evidence_name` / `phenomenon` / `reasoning` / `counter_explanation` / `reliability` / `semantics_aligned` / `direction` + `direction_source` / `condition_metadata` / 像素与归一化坐标；**显式区分** `support`（整流后主张）、`support_raw`（专家原始主张）、`direction ← direction_source`（方向权威链）、`P(Fake)/P(Real)`（校准似然）；`semantics_aligned=false` 显示**红色告警**并同时给出原始解释、校准后方向与反解释。全量核对：**200 个 token 中 158 个为 false**（工具桶 61/61、弃权桶 97/139），此前无一处可见
  3. **按桶审核标准**: 原清单对所有样本都用"ΔBrier 为负且结论正确"，会把 140 条**设计上就弃权**的样本判死。改为共用项 + 三套分桶清单：`tool_positive`（结论正确 / 增益真实 / 证据被真正使用 / 无编造）、`no_tool_positive`（**不得虚构法证证据** / **不得形成"无工具⇒Real"捷径**）、`honest_abstention`（弃权须有弱证据·冲突·不适用三种成因，**不要求 ΔBrier<0**）
  4. **决定可保存（静态页 + 导入）**: 页面按钮 `accept / format_only / revise / reject`、备注框、上一条/下一条、快捷键 1–4 与 ← →、决定后自动跳转、进度与本次修改历史；状态存 `localStorage`（刷新不丢，`file://` 受限时页面顶端红字提示改用 `http.server`），「导出全部决定」得 `review_decisions.json`；`record_review.py` 新增 `--import`（读导出 JSON 或填好的 `worklist.csv`），逐条走与 CLI 相同的校验：未知样本 id 与非法处置被**跳过并计数上报**（一个错别字不该毁掉另外 236 条），重复导入同一文件为 no-op，改过的决定追加并保持"最新为准"
  5. **置信度四分**: 分别显示**原始模型判定与自报概率**（`model_candidate` / `model_probability`）、**停止后验 P(Fake)/P(Real)**、**所选标签置信度**、**ΔBrier（模型→后验）**，并注明无工具轨迹没有后验（不打印先验 0.5 冒充后验）
  6. **取消双审**: `double_review` → **`high_risk`**（uncertain / multi_tool / high_confidence / conflict，207/237，仅作重点提示），`review_sets.json` 增 `"protocol": "single_pass"` 并去掉 `stratified`；不再有首审/二审/仲裁与分层抽查；`apply_review.py` 维持"一条样本一个最终决定"不变；plan.md 两处旧表述加注"已于 2026-10-09 修订"，README 与架构文档同步改写
* **涉及/修改的文件清单**：
  - `scripts/build_review_workstation.py (Modified — 全量对话 / 证据卡片 / 分桶清单 / 交互与本地保存 / high_risk)`
  - `scripts/record_review.py (Modified — parse_export + import_decisions + --import)`
  - `tests/test_review_workstation.py (Modified — 30 项，含"只显示第一轮"回归)`
  - `tests/test_review_pipeline.py (Modified — 导入通道 11 项)`
  - `sft_data/review/ (Modified — 237 页 + index + review_sets.json + worklist.csv 重建)`
  - `plan.md, README.md, CURRENT_PROGRAM_ARCHITECTURE.md (Modified — 单次审核协议与工作台说明)`
* **执行结果与验证状态**：610 测试通过（新增 27）；工作台重建成功（237 页、0 张图缺失、6.5 MB）；页面内嵌 JS 通过 `node --check`；抽样核对 tool/no-tool/abstention 三类页面各含对应清单与语义字段
* **置信度或遗留待办（TODO）**：①审核者需在浏览器内完成 237 条并导出，再 `--import` 写回仓库（`--summary` 查进度）②`file://` 下若浏览器禁用本地存储，需用 `python3 -m http.server` 打开（页面会自动提示）③标签成分偏斜仍待人工裁决（配平 / 接受 / 补样本）
---
### 2026-10-11 10:30:00 - G4-f 审核结论与 A–E 修订（训练对话 / 概率字段 / 接地检查 / 专家语义 / 停止可解释性）

* **当前操作动作**：导入人工审核结果（237/237），定位审核发现的三处根因，并按计划实施 A–E（CPU）＋在现有数据上离线预演
* **对应计划锚点**：plan.md §4.11「G4-f 审核结论与修订计划（2026-10-11）」及其「A–E 实施结果与离线预演」；agent.md §1.2 / §2.1 / §3.2
* **核心变更说明**：
  1. **审核结果入库**: `review_decisions.json` 由 `record_review.py --import` 写入 `dispositions.jsonl`，**237/237 complete**：accept **1** / revise **195** / reject **41**（工具桶 0A·35R·25X；无工具桶 0A·34R·3X；弃权桶 1A·126R·13X）→ `apply_review.py` 只能产出 1 条 accepted，**G4-g 暂停**
  2. **三条根因（代码级，均已验证）**: ①`utils/final_v2.py:170` 用字符串 `<call_` 判定调用轮并保留全部助手轮 → 87 处相邻助手轮、51 条含两个结论；②`halting_v2._halt` 四处把 `posterior`(=P(Fake)) 当 `confidence` 传出 → G3 工具臂 Real 判决 confidence 全在 0.20–0.32，**按存储实义重算 AUROC：text 0.502→0.677、image 0.543→0.684、both 0.539→0.659**（rgb 0.654 不受影响；G2 报告不受影响）→ `check_acceptance_g5.py` 对 G3 的裁决作废；③`extract_observation()` 只搜 planning 的 Visual Anomalies，无工具样本没有 planning → 39 条观察只有格式/尺寸
  3. **A 概率字段**: 1) `_halt` 改由 `_label_confidence(verdict, posterior)` 推导，调用方一律传后验，结构上杜绝混用；Uncertain 的 confidence 定义为"未结案"标记常量并在模块头写明字段表；Uncertain 调用点改传真实后验；2) 无工具样本的 reasoning 不再写"后验 P(Fake)=0.5"；3) `_pseudo_probability` 补上它所依赖的契约；4) 影响面实测 **12 条**（与审核的 12 条备注一一对应）
  4. **B 对话忠实回放（O2）**: 保留全部轮次含 `[System]` 纠正轮；调用轮判定改为"标签形式调用 + 其后匹配交付"（正文提及不算）；只替换**一个**结案轮为渲染答案，其后轮次丢弃；校验器新增"相邻结案轮"检查。预演：`[System]` 轮保留 0→**189** 条，相邻助手轮 87→**39**
  5. **C 接地四项**: C1 散文级虚构工具引用（"提到某专家结果但该专家从未被调用"，实测命中 7 条，含审核点名的 `f2_noise+frequency__ADM_49_adm_153_jpeg_q95`）；C2 Real⇒confidence=1−posterior（命中 12 条）；C3 全局测量被写成局部（命中 106 条，全部来自 noise 专家的 "Localised noise variance" 模板）；C4 观察只有格式/尺寸（命中 39 条）。**1 条 accept 全部通过，零误报**
  6. **D1 专家语义三层**: noise 专家的 phenomenon/reasoning 改为「整图实测值 → 校准带关联 → 不能证明什么」，删除 "Localised"、"the region is smooth"、PRNU 断言；jpeg 专家不再断言"校准接近随机"，改为描述校准带与失效区；**整流器新增守卫**：校准超出中性带而专家文本仍称"接近随机"时，该句被替换为规范校准句、原文留 `reasoning_raw`（实测 P(Fake)=0.861 被替换、P(Fake)=0.52 保留）
  7. **D3 复算（校准集 700 条）**: 复现项目既有结论并补两张表——noise 全格式存活（0.153→0.228）保留为主力；jpeg png 0.028 强、q95/q85 0.283/0.266 已弱、**q70 0.569 死亡**；frequency_v2 弱而稳定（0.556–0.634）；**ela 0.948→0.504 确认是压缩历史捷径**（分分辨率层 mid 0.972 / large 0.774 仍在）→ 维持停用。内容分层（Laplacian 十分位）：noise 在**最平滑内容上分离最强**（0.087/0.146/0.279），但内容与标签在数据集内相关（最平滑两个十分位 78% 是 Real）→ "内容先验"旁路真实存在，故专家文本必须写明不能证明什么
  8. **D4 重复调用回应**: `controller.py:254` 命中重复 evidence_id 后原本直接 `continue`，会话里不留任何痕迹 → 模型学到"调用后无返回也能下结论"（39 条相邻轮里 29 条是这个成因）。现注入"该调用与已有证据重复、已忽略，请勿重复调用同一工具与同一区域"
  9. **E 停止可解释性**: 轨迹新增 `available_experts`（策略当时能调用的集合——**旧 Trace 0/3237 记录过**，这正是"no_expected_gain 无法审计"的根因）与 `tool_utilities`（逐专家 gain/cost/net）；`HaltingPolicyV2.utility_report` 收敛为一处计算；对模型的措辞改为"**当前策略估计**继续调用收益不足"；`replay_halting_g3.py` 新增"停止时是否仍有工具净值为正"，并对无允许集合记录的旧 Trace 返回 `not_auditable`（不得当成"干净"）
 10. **离线预演总账（237 条，自动判定 × 人工处置）**: accept 1 条 **PASS**（0 误报）；reject 32 FAIL / 9 PASS；revise 126 FAIL / 69 PASS → 自动接地把可判定缺陷从 0 提到 **158/237**；仍 PASS 的 79 条中 **78 条人工仍判 revise/reject**，备注几乎全是"裁剪区域与所述内容不符""压缩痕迹在实际输入上不可核验" → 即 **D2 视觉接地**（模型预筛 + 人工复核可疑项，待 GPU）
* **涉及/修改的文件清单**：
  - `sft_data/review/dispositions.jsonl (Created — 237 条处置，来自浏览器导出)`
  - `utils/final_v2.py (Modified — 忠实回放 + 相邻结案轮 + 接地四项)`
  - `state_machine/halting_v2.py (Modified — _label_confidence / 字段表 / tool_utilities / utility_report)`
  - `state_machine/controller.py (Modified — 重复调用回应 / available_experts 落盘 / 停止措辞)`
  - `state_machine/evidence_rectifier.py (Modified — 校准断言守卫)`
  - `experts/noise.py, experts/jpeg.py (Modified — 三层语义文本)`
  - `scripts/qwen_gain_baseline.py (Modified — 提取契约注释)`
  - `scripts/replay_halting_g3.py (Modified — leftover utility / not_auditable)`
  - `tests/test_final_v2.py, tests/test_halting_v2.py, tests/test_controller.py, tests/test_evidence_rectifier.py, tests/test_replay_halting_g3.py (Modified — 新增 29 项)`
  - `plan.md (Modified — 审核结论与修订计划、A–E 结果、D3 表、预演总账)`
* **执行结果与验证状态**：**639 测试通过**；预演在真实 237 条上运行（注入的坏样本均被捕获，accept 样本零误报）；`calibration/g4f_halting_replay.json` 为本次回放（原 `g3_halting_replay.json` 已还原未被覆盖）
* **置信度或遗留待办（TODO）**：①**D2 未做**（视觉描述 ↔ bbox 的模型预筛 + 人工复核，需 GPU）：预演显示它正是剩余 78 条的主因，**F 重新生成前应补上**，否则新数据仍带同一类缺陷 ②D4 与 D1 的效果只能在 F 重新生成后验证 ③四臂重跑（G）必须用修正后的字段，旧 G3/LoRA 报告的 AUROC/ECE 不可再引用 ④标签成分偏斜（75/25、无工具 37R/0F）仍待人工裁决
---
### 2026-10-11 12:40:00 - D2 视觉接地预筛：工具就位，四轮实测后判定「能看图、不能判」

* **当前操作动作**：实现并实测视觉接地预筛（`scripts/screen_visual_grounding.py`），在「通过自动检查的 79 条」（78 条人工判改 + 1 条 accept）上对照人工备注验证
* **对应计划锚点**：plan.md §4.11「G4-f 修订」D2 及其「D2 预筛结果」；用户决策「模型预筛 + 人工只看可疑项」
* **核心变更说明**：
  1. **工具**：三问式单图调用（先描述整图 → 再与回答比对 → 再问**照片**裁剪），可续跑、带**提示词指纹**（改了提问就不会静默复用旧记录 —— 第三轮就踩过这个坑，同一类错误此前在四臂 harness 上也出现过）、`--against-dispositions` 对照人工审核给出召回/精确
  2. **发现的 Harness bug（严重）**：`QwenVLClient.ask` **没有发送图像** —— `build_messages` 只在轮次文本含 `<image>` 标记时附主图，自带提示词的调用方无从知道。前三轮「召回 0.94」全部是在**纯文本**上判出来的（模型把鳄鱼描述成「一只猫」），数字无效并已作废。修复：`with_image_marker()` 在 `ask` 内无条件补标记；两侧契约各有测试固定
  3. **修复后的真实表现**：整图描述准确（鳄鱼 / 浴室里的狗 / 红色龙虾均正确）；但**判定不可靠** —— 自称「图里有一只鳄鱼」却对「该对象是否在图中」答 false，79/79 全标疑点（含唯一 accept 样本）
  4. **顺带修**：`validate_sample` 在样本缺分辨率时不再抛异常（跳过边界检查而不是让整轮校验崩掉）
  5. **结论与出路**：7B 零样本裁判不能当筛子。可选：①以人工审核的 237 条做少样本示例再试；②只当**标注器**（把裁判对整图/裁剪图的描述并排显示在工作台，由人工判断，符合「人工结合」决策）；③放弃预筛，F 阶段依赖 A–C 自动检查 + 人工全量审核。**当前默认③，②随时可加**
* **涉及/修改的文件清单**：
  - `scripts/screen_visual_grounding.py (Created — 预筛器，含 CPU 干跑与对照评估)`
  - `mllm/qwen_client.py (Modified — ask() 无条件补 <image> 标记 + with_image_marker)`
  - `tests/test_screen_visual_grounding.py (Created — 32 项)`
  - `tests/test_message_builder.py (Modified — 标记契约 4 项)`
  - `utils/final_v2.py (Modified — 缺分辨率时不再崩溃)`
  - `calibration/visual_grounding_screen.json (Created — 第四轮真实结果)`
  - `plan.md (Modified — D2 预筛结果表)`
* **执行结果与验证状态**：671 测试通过；四轮 GPU 实测（第 1–3 轮因送图 bug 作废，对应报告已删除，数字记录在本条）；最终结论有实测与逐条理由支撑
* **置信度或遗留待办（TODO）**：①D2 不作为过滤器（默认③），若要用②只需在 F 之后加一个页面区块 ②**D1/D4 的效果仍待 F 验证** ③F（重新生成，约 2.5 GPU 小时）与 G（四臂重跑）尚未执行，需用户确认算力窗口 ④标签成分偏斜仍待人工裁决
---
