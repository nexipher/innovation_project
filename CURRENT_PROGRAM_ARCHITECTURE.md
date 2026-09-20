# Forensic-Agent 当前程序框架与运行逻辑

本文档描述当前仓库中已经实现的程序结构和单次图像分析流程，主要面向人工代码审计、项目交接和后续开发。内容以当前源码为准，不代表 `plan.md` 中尚未实现的 LoRA、GRPO 和全数据集评估功能。

## 1. 程序定位

Forensic-Agent 是一个由多模态大语言模型（MLLM）主动调度法证专家的图像真实性检测原型。

系统将 MLLM 作为高层“法官”：模型先观察图像、定位可疑区域，再通过结构化动作标签调用频域、噪声或 JPEG 专家。专家计算得到的物理指标会被转换为 Evidence Token，并重新注入模型上下文。模型结合视觉观察和物理证据进行多轮推理，最终输出 `Real`、`Fake` 或 `Uncertain` 判定及法证报告。

当前系统采用纯 Python `while` 循环实现状态机，不依赖 LangChain 等通用 Agent 框架。

## 2. 整体框架

```mermaid
flowchart TD
    CLI["main.py<br/>解析命令行参数"] --> BUILD["build_pipeline()<br/>选择 Mock 或 Qwen"]
    BUILD --> FSM["ForensicStateMachine.run()"]
    FSM --> LOAD["加载图像<br/>初始化 SessionLogger"]
    LOAD --> MLLM["MLLM.generate()<br/>生成 planning / call / verdict"]
    MLLM --> PARSER["Parser<br/>解析 XML 风格标签"]

    PARSER -->|存在 verdict| FINAL["解析最终判定"]
    PARSER -->|存在 call_*| COORD["坐标 [0,1000] 转像素坐标"]
    PARSER -->|格式不合法| REPAIR["注入格式纠错消息"]
    REPAIR --> MLLM

    COORD --> CROP["裁剪 bbox"]
    CROP --> EXPERT["Frequency / Noise / JPEG Expert"]
    EXPERT --> TOKEN["EvidenceTokenizer<br/>生成 Evidence Token"]
    TOKEN --> HISTORY["写入运行历史和证据链"]
    HISTORY --> HALT["HaltingChecker"]

    HALT -->|继续探索| MLLM
    HALT -->|冲突 / 收敛 / 预算耗尽| FORCE["要求 MLLM 强制结案"]
    FORCE --> FINAL
    FINAL --> SAVE["SessionLogger<br/>保存 ShareGPT JSON"]
```

## 3. 分层结构

| 层次 | 主要文件 | 核心职责 |
|------|----------|----------|
| 入口层 | `main.py` | 解析 CLI 参数，构造组件，执行单张或批量分析 |
| 配置层 | `config.py` | 保存路径、阈值、最大步数、专家参数和 System Prompt |
| 模型层 | `mllm/base.py`、`mock_client.py`、`qwen_client.py` | 提供统一 MLLM 接口，并实现 Mock 与真实 Qwen 后端 |
| 控制层 | `state_machine/controller.py` | 编排“生成—解析—调用专家—回灌证据—终止”的主循环 |
| 解析层 | `utils/parser.py` | 解析 `<planning>`、`<call_*>`、`<reasoning>` 和 `<verdict>` |
| 专家层 | `experts/` | 从图像局部区域提取频域、噪声和压缩痕迹 |
| 抽象层 | `state_machine/evidence_tokenizer.py` | 将专家结果转换为统一 Evidence Token JSON |
| 终止层 | `state_machine/halting.py` | 检查主动结案、最大步数、证据冲突和信息增益收敛 |
| 数据层 | `utils/logger.py` | 保存对话、证据链、判定结果和 SFT Trace |

## 4. 核心组件

### 4.1 CLI 与管道构建

`main.py` 是程序入口，支持以下主要参数：

```text
--image / -i    分析一张图像
--batch / -b    按数据集类别批量分析
--mllm          选择 mock 或 qwen
--mode / -m     选择 Mock 行为模式
--max / -n      限制每类批量图像数量
```

`build_pipeline()` 根据 `--mllm` 创建模型客户端，并注册三个专家：

```python
{
    "frequency_expert": FrequencyExpert(),
    "noise_expert": NoiseExpert(),
    "jpeg_expert": JPEGExpert(),
}
```

当前 CLI 主管道使用 `FrequencyExpert` v1。`experts/frequency_v2.py` 已存在，但主要用于阶段三的数据构造和实验，尚未接入 `main.py` 的默认分析管道。

### 4.2 MLLM 抽象层

所有模型客户端继承 `BaseMLLMClient`，统一暴露：

```python
generate(image_path, history) -> str
reset() -> None
name -> str
mode -> str
```

当前有两种实现：

- `MockMLLMClient`：根据模板和轮次生成结构化响应，用于 CPU 开发和状态机测试；它不具备真实视觉理解能力。
- `QwenVLClient`：加载 Qwen2.5-VL-7B-Instruct，在 GPU 上执行真实多模态推理，并在输出格式不合法时最多重试两次。

当用户选择 `--mllm qwen` 但 CUDA 不可用时，`main.py` 当前会打印警告并自动降级为 Mock。

### 4.3 输出协议与解析器

MLLM 使用 XML 风格标签与状态机通信：

```xml
<planning>
Suspected Region: [200, 200, 800, 800]
Visual Anomalies: 描述可疑视觉现象
Expert Target & Hypothesis: 说明专家选择及假设
</planning>
<call_noise>[200, 200, 800, 800]</call_noise>
```

获得足够证据后，模型应输出：

```xml
<reasoning>
结合视觉观察和 Evidence Token 进行物理—语义一致性分析。
</reasoning>
<verdict>
{"verdict":"Fake","confidence":0.86,"primary_evidence":["noise_residual_inconsistency"],"report":"..."}
</verdict>
```

`Parser` 使用正则表达式提取标签，并对 `call_frequency`、`call_call_noise` 等常见模型标签错误进行名称归一化。`<verdict>` 内部必须能够解析为 JSON，状态机才会将其视为有效结论。

### 4.4 法证专家

三个专家都继承 `BaseExpert`，接收状态机已经裁剪好的 BGR 图像 patch，并返回 `ExpertResult`。

| 专家 | 主要算法 | 目标现象 |
|------|----------|----------|
| Frequency | Hanning 窗、2D FFT、高频局部峰值 | GAN/Diffusion 上采样产生的周期性频域结构 |
| Noise | SRM 高通残差、局部方差图、不一致性度量 | 拼接、重绘或平滑造成的微观噪声断层 |
| JPEG | 8×8 块边界梯度、DCT 系数直方图 | 块效应、双重量化和二次保存痕迹 |

`ExpertResult` 主要包含：

- `evidence_name`：证据名称；
- `phenomenon`：实际检测到的物理现象；
- `reasoning`：该现象的法证解释；
- `strength`：范围为 `[0,1]` 的异常强度；
- `source`：专家来源；
- `support`：`Real`、`Uncertain` 或 `AI-generated`；
- `raw_metric` 和 `metadata`：算法调试信息。

### 4.5 Evidence Token

`EvidenceTokenizer` 将 `ExpertResult` 转换成可注入 MLLM 上下文的 JSON：

```json
{
  "evidence_name": "noise_residual_inconsistency",
  "region": "patch_coordinates_[100, 120, 500, 520]",
  "phenomenon": "Localized noise variance measures abnormally...",
  "reasoning": "Significant localized noise variance anomaly detected...",
  "strength": 0.763,
  "source": "noise_expert",
  "support": "AI-generated",
  "interpretation_text": "Severe statistical anomaly matching artificial generative fingerprints."
}
```

默认强度映射为：

| strength | support | 含义 |
|----------|---------|------|
| `0.0 <= s < 0.3` | Real | 与正常相机成像统计特征一致 |
| `0.3 <= s < 0.7` | Uncertain | 存在轻微或不确定异常 |
| `0.7 <= s <= 1.0` | AI-generated | 存在显著人工生成特征 |

## 5. 单次分析的完整运行逻辑

### 5.1 启动与初始化

单张分析示例：

```bash
python main.py --image dataset/GenImage_Test/Midjourney/example.png --mllm mock
```

程序首先验证文件是否存在，然后从路径推断 ground truth：路径包含 `/Real/` 时标记为 `Real`，其他路径默认标记为 `Fake`。该标签只应作为数据集评估和 Trace 元数据使用，不应被真实检测逻辑当作输入证据。

随后创建 MLLM、三个专家和 `ForensicStateMachine`，并调用：

```python
fsm.run(image_path, ground_truth=gt)
```

状态机开始时会：

1. 调用 `mllm.reset()`；
2. 通过 OpenCV 加载图像；
3. 获取图像高度和宽度；
4. 初始化 `SessionLogger`；
5. 创建空的运行时对话历史和证据链；
6. 将步数设为零，进入主循环。

### 5.2 第一轮 MLLM 决策

状态机调用：

```python
raw_output = mllm.generate(image_path, conversation)
```

模型可以选择：

- 输出 `<planning>` 和一个或多个 `<call_*>` 请求证据；
- 输出 `<reasoning>` 和 `<verdict>` 直接结案；
- 输出不合法内容，由状态机注入纠错消息后重试。

状态机优先解析 verdict。只要输出中存在可解析且包含 `verdict` 字段的 `<verdict>`，循环立即结束；同一响应中即使还包含 `<call_*>`，这些调用也不会再执行。

### 5.3 专家调度

当模型输出合法调用时，状态机按以下流程处理每个调用：

1. `Parser.extract_all_calls()` 提取专家名称和相对 bbox；
2. `CoordinateTransformer.relative_to_absolute()` 将 `[0,1000]` 坐标转换为图像像素；
3. `clip_bbox()` 将坐标限制到图像边界内，并保证最小裁剪尺寸；
4. `ImageUtils.crop_bbox()` 裁剪图像 patch；
5. 根据调用名称查找对应专家；
6. 执行 `expert.analyze(patch)`；
7. 将 `ExpertResult` 转换为 Evidence Token；
8. 把 Evidence Token 写入证据链，并作为新的 user 消息注入对话历史。

一轮 MLLM 输出可以包含多个专家调用。当前 `step` 在处理完该轮所有调用后只增加一次，因此它表示“包含专家调用的模型轮数”，不严格等于专家调用总次数。

### 5.4 终止判断

每轮专家调用完成后，`HaltingChecker.check()` 按优先级检查：

1. **模型主动结案**：输出了有效 `<verdict>`；
2. **最大步数**：`step >= MAX_STEPS`，当前默认上限为 5；
3. **证据冲突**：证据链中同时存在 `strength > 0.7` 和 `strength < 0.3`；
4. **信息增益收敛**：最后两条证据的 strength 差值小于当前阈值。

最大步数或信息增益收敛时，状态机会追加预算耗尽提示，再调用一次 MLLM 生成最终 verdict。证据冲突时，则追加“疑罪从无”提示，要求模型进行双向反思并输出 `Uncertain`。

如果最终输出仍无法解析，状态机会生成兜底判定：

```json
{
  "verdict": "Uncertain",
  "confidence": 0.5,
  "report": "取证资源耗尽或分析过程异常终止。"
}
```

### 5.5 Trace 保存与返回

循环结束后，`SessionLogger` 将完整会话保存为 ShareGPT 风格 JSON，内容包括：

- Session ID 和图像路径；
- ground truth 与图像来源；
- 全部 user/gpt 对话；
- Evidence Token 链；
- 最终 verdict；
- 图像尺寸、总步数、终止原因和模型模式。

文件写入：

```text
traces/sft_sessions/forensic_sft_session_时间_图像名.json
```

状态机同时向调用者返回内存中的结果字典，`main.py` 据此打印 verdict、confidence、步数、终止原因、专家证据和 SFT 文件路径。

## 6. 运行时序示例

```mermaid
sequenceDiagram
    participant CLI as main.py
    participant FSM as ForensicStateMachine
    participant M as MLLM
    participant P as Parser
    participant E as Forensic Expert
    participant L as SessionLogger

    CLI->>FSM: run(image_path, ground_truth)
    FSM->>L: init_sft_session()
    FSM->>M: generate(image, history=[])
    M-->>FSM: planning + call_noise[bbox]
    FSM->>P: extract_all_calls()
    P-->>FSM: noise + relative bbox
    FSM->>E: analyze(cropped patch)
    E-->>FSM: ExpertResult
    FSM->>L: add_evidence(Evidence Token)
    FSM->>M: generate(image, history+evidence)
    M-->>FSM: reasoning + verdict JSON
    FSM->>P: parse_verdict()
    FSM->>L: finalize_sft() + save_sft()
    FSM-->>CLI: session result
```

## 7. 当前实现边界

- Mock MLLM 用于验证控制流，不能代表真实检测准确率；
- 真实 Qwen2.5-VL 需要 CUDA 和约 16 GB FP16 显存；
- 当前默认主管道仍使用 Frequency Expert v1；
- Noise/JPEG 信号会受到 Real JPEG、Fake PNG 数据格式差异影响；
- LoRA 微调、GRPO、全数据集评估和消融实验尚未进入当前运行管道；
- 每次运行都会生成一份 Trace，可用于调试和后续 SFT 数据加工。

## 8. 主要代码阅读入口

建议按以下顺序阅读源码：

1. `main.py`：理解外部入口和组件装配；
2. `state_machine/controller.py`：理解完整控制流；
3. `utils/parser.py`：理解 MLLM 与程序的通信协议；
4. `experts/base.py` 和三个专家实现：理解物理证据来源；
5. `state_machine/evidence_tokenizer.py`：理解证据抽象；
6. `state_machine/halting.py`：理解循环退出条件；
7. `mllm/mock_client.py` 与 `mllm/qwen_client.py`：比较测试后端和真实后端；
8. `utils/logger.py`：理解 Trace 和 SFT 数据落盘方式。

## 9. 新增参考论文与项目借鉴分析

本节保留此前第一组参考论文的对照结论。原始 PDF 当前已不在 `ref/` 目录，以下内容作为当时已完成核验的分析记录：

1. *From Pixels to Semantics: A Novel MLLM-Driven Approach for Explainable Tampered Text Detection*，ACM MM 2025，提出 TVSIP；
2. *Propose and Rectify: A Forensics-Driven MLLM Framework for Image Manipulation Localization*，IEEE TIFS 2026，提出 Propose-Rectify。

两篇论文的任务都包含图像篡改定位，而当前项目主要处理 Real/Fake/Uncertain 图像级真实性判断。因此，应借鉴它们的系统原则和验证方法，不宜把文本 OCR 或像素级分割模块不加区分地直接移植到当前代码中。

### 9.1 与当前工作的总体关系

| 维度 | 当前 Forensic-Agent | TVSIP | Propose-Rectify |
|------|----------------------|-------|-----------------|
| MLLM 角色 | 主动选择专家并生成最终 verdict | 语义定位分支 + 解释器 | 提出初始假设，不直接拥有最终裁决权 |
| 低层证据 | 三个独立专家输出标量与文本 | 专家像素掩码 | SRM、Bayar、Sobel、Noiseprint++ 特征图 |
| 融合方式 | Evidence Token 文本回灌 | 视觉掩码与语义框相加 | 分析引导门控 + 多尺度交叉注意力纠偏 |
| 空间输出 | MLLM bbox，专家分析局部 patch | OCR 校正 bbox + 像素掩码 | SAM 输出精细篡改掩码 |
| 解释约束 | reasoning 引用 Evidence Token | 对最终高亮区域进行解释 | 最终输出由法证特征纠偏后的表示产生 |
| 当前最值得借鉴 | — | 语义分支、定位—解释对齐、退化评估 | 提案—纠偏职责划分、多特征自适应验证 |

这两篇论文共同支持当前项目的基本出发点：MLLM 的语义能力和传统法证特征是互补的，单独依赖任一分支都不够可靠。不过，它们也指出当前设计的一个关键不足：专家证据不应只作为供 MLLM 自由解读的附加文本，还应对初始判断形成可验证、可校准的纠偏约束。

### 9.2 TVSIP 可以带来的启发

#### 9.2.1 将“语义异常”显式建模为第四类证据

TVSIP 的 Locator 包含低层视觉线索分支（LLVCB）和高层语义线索分支（HLSCB）。论文在第 4-5 页将 HLSCB 拆为检测、位置描述和 bbox 提取三个连续任务，再与专家掩码融合。其消融实验显示，HLSCB 单独定位能力有限，但与多个不同专家模型组合时都能提高结果；在 JPEG、缩放、噪声和模糊退化下，语义线索还能补偿低层痕迹衰减（第 7 页，表 4、表 6、表 7）。

当前项目的 `planning` 已包含视觉异常描述，但这部分没有形成独立、可审计的结构化证据。可以增加 `semantic_observation` Evidence Token，例如：

```json
{
  "evidence_name": "semantic_physical_inconsistency",
  "region": "patch_coordinates_[...]",
  "phenomenon": "主体阴影方向与场景主光源不一致",
  "reasoning": "该现象可能来自生成或局部合成，也可能由多光源环境造成",
  "strength": 0.58,
  "source": "semantic_expert",
  "support": "Uncertain"
}
```

这样可以把“MLLM 看到了什么”和“传统专家测到了什么”放入同一证据链，而不是让语义观察只存在于不可统计的自由文本中。

#### 9.2.2 分离定位器与解释器，避免定位—报告错位

TVSIP 明确区分 Locator 和 Interpreter，并要求 Interpreter 同时查看原图和标出最终可疑区域的高亮图。论文第 5 页和第 8 页的实验表明，相比仅提供非融合掩码或不提供 Locator，最终融合区域有助于检测、定位和解释保持一致。

当前项目可以先进行不依赖新模型的轻量改造：

1. 状态机汇总所有专家调用 bbox，生成一张半透明高亮图；
2. 最终结案轮同时向 Qwen 提供原图和高亮图；
3. verdict 中新增 `regions` 与 `evidence_ids`，要求每个报告结论绑定区域和证据；
4. 增加自动一致性检查：报告引用的区域必须与实际调用 bbox 具有足够 IoU。

这比仅在文本中写 `patch_coordinates_[...]` 更能帮助 MLLM 保留空间上下文，也更适合后续实现 Attention-Evidence Consistency Reward。

#### 9.2.3 借鉴框校正思想，但不直接照搬 OCR

TVSIP 使用 OCR 文本候选校正 MLLM 漂移的 bbox。这对文档篡改十分有效，但当前项目处理一般自然图像，不能直接依赖 OCR。

可迁移的是“MLLM 粗定位 + 外部结构校正”的原则。一般图像可以考虑：

- 使用 SAM 或边缘/显著性区域把粗 bbox 吸附到物体或异常边界；
- 将 bbox 扩展为多尺度区域：局部、上下文环带和全图；
- 对过小、越界、极端长宽比区域进行确定性校正；
- 在 Trace 中同时保存原始 bbox 与校正后 bbox，便于审计定位漂移。

#### 9.2.4 改进 SFT 数据结构和质量控制

TVSIP 的 TextDDLE 将输出拆成 Description、Detection、Localization、Explanation 四项，并采用 GPT-4o 生成后人工筛查；其训练采用大量合成数据预训练，再用较少真实困难样本进行 SFT（第 3-5 页）。解释又进一步拆为低层视觉线索和高层语义线索。

对当前 539 条候选训练数据及 38 条拒绝记录，可借鉴以下结构：

- 在训练响应中显式分开 `visual_observation`、`forensic_evidence`、`contamination_analysis` 和 `conclusion`；
- 训练样本同时保留“无语义异常”和“无低层异常”的合法情况，避免模型强行编造两类证据；
- 对自动生成的 reasoning 做人工抽样审核，重点删除不存在的视觉现象和错误因果关系；
- 未来扩充数据时采用“廉价合成格式训练 → 高质量真实推理 SFT”的两阶段策略。

### 9.3 Propose-Rectify 可以带来的启发

#### 9.3.1 将 MLLM 从最终法官调整为假设提出者

Propose-Rectify 最重要的原则是：MLLM 负责提出初始分析和可疑区域，最终判断必须接受正交法证证据的系统纠偏。论文强调“refinement”只是把已经正确的预测做精，而“rectification”需要验证初始推理是否可能错误（第 4 页）。

这与当前系统非常接近，但权责不同：当前专家返回 Evidence Token 后，最终 verdict 仍完全由 MLLM 自由生成。更稳健的演进方式是增加独立的 `EvidenceRectifier`：

```text
MLLM Proposal
    → 专家证据与可靠性校准
    → RectificationDecision
    → MLLM 仅负责把纠偏结果解释成人类可读报告
```

`RectificationDecision` 至少应包含：

```json
{
  "proposal": "Fake",
  "rectified_label": "Uncertain",
  "confidence": 0.54,
  "supporting_evidence": ["E1"],
  "contradicting_evidence": ["E2"],
  "reliability_context": {
    "image_format": "PNG",
    "compression_detected": false
  },
  "reason": "频域信号较弱，噪声证据受图像格式混杂影响"
}
```

短期可以用规则和校准概率实现，不必立即训练神经网络。这样能防止模型无视低强度或冲突证据，直接输出高置信度结论。

#### 9.3.2 从离散专家调用升级为分析引导的证据路由

论文的 Analysis-Informed Feature Gating 让 MLLM 分析表示查询法证特征，并分别为局部、中尺度和全局特征生成权重（第 6 页）。消融中移除门控后，检测 F1 从 0.809 降至 0.750，定位 F1 从 0.423 降至 0.360（第 11 页，表 5-6）。

当前 `<call_freq|noise|jpeg>` 已经是离散版门控，但缺少可靠性建模。可以把路由从“调用/不调用”扩展为：

- 根据图像格式、尺寸、压缩程度和 MLLM 假设计算每个专家的适用度；
- 记录 `requested_reason`、`expected_signal` 和 `reliability_weight`；
- 禁止把不同物理含义的 strength 当作可直接比较的同尺度概率；
- 使用校准后的证据似然或置信区间进行融合，而不是只使用统一的 0.3/0.7 阈值。

这能直接缓解当前 Real=JPEG、Fake=PNG 造成的 noise/jpeg 格式混杂问题。

#### 9.3.3 专家应输出多尺度空间特征，而不只是单个标量

Propose-Rectify 同时使用 SRM、Bayar、Sobel 和 Noiseprint++，并在局部、中尺度和全局三个尺度上逐步纠偏语义表示（第 4-6 页）。当前系统的三个专家主要把一个 patch 压缩为单个 `strength`，会丢失异常究竟位于 patch 内部何处的信息。

建议逐步扩展 `ExpertResult`：

```text
现有：strength + phenomenon + reasoning
下一步：增加 raw_metric、reliability、scale_scores
再下一步：增加 heatmap_path / mask_path / region_statistics
长期：保留可学习的 feature map，进入跨注意力纠偏模块
```

其中 Sobel 边界和多尺度 SRM 可以先用 CPU 实现；Bayar 约束卷积和 Noiseprint++ 属于需要训练或加载权重的后续模块。

#### 9.3.4 像素级定位是合理的长期方向

论文通过 Enhanced Segmentation Module 对齐 SAM 语义特征和法证特征，并显式放大二者的差值与乘积响应（第 7 页）。在消融实验中，移除该模块后定位 F1 从 0.423 降至 0.403、IoU 从 0.351 降至 0.334（第 11 页）。

当前项目尚不具备像素级 GT，也主要面向 AI 生成图像的全局判断，因此现在直接训练 SAM 分支成本较高。更合适的顺序是：

1. 先保存专家 heatmap 和高亮 bbox；
2. 在具有局部篡改 mask 的数据集上单独验证定位能力；
3. 再引入 SAM 解码器和分割损失；
4. 最终联合优化图像级检测、区域级证据一致性和像素级定位。

### 9.4 推荐的项目演进路径

```mermaid
flowchart LR
    A["当前系统<br/>MLLM 调用专家<br/>标量 Evidence Token"] --> B["阶段 A：规则纠偏<br/>EvidenceRectifier<br/>格式/退化可靠性"]
    B --> C["阶段 B：空间证据<br/>多尺度分数<br/>heatmap + 高亮图"]
    C --> D["阶段 C：自适应路由<br/>分析引导专家权重<br/>概率校准"]
    D --> E["阶段 D：学习式纠偏<br/>法证特征图融合<br/>跨注意力"]
    E --> F["阶段 E：精细定位<br/>SAM/分割头<br/>多任务训练"]
```

建议优先级如下：

| 优先级 | 建议 | 原因 | GPU |
|--------|------|------|-----|
| P0 | 修复 Qwen 多轮图像历史并引入 `EvidenceRectifier` | 保证主管道逻辑正确，防止 MLLM 无视证据 | 否 |
| P0 | 为专家增加格式/退化条件下的可靠性字段 | 直接处理当前 JPEG/PNG 混杂 | 否 |
| P1 | 保存专家 raw metric、多尺度分数和 heatmap | 避免空间证据被压缩成单个标量 | 否/可选 |
| P1 | 最终轮输入原图 + 可疑区域高亮图 | 提高定位—解释一致性 | Qwen 推理需要 |
| P1 | 扩充 SFT Schema 并抽样人工复核 | 降低自动 reasoning 幻觉和错误因果 | 否 |
| P2 | 加入 Sobel/Bayar/Noiseprint++ 并学习门控 | 获得更强、多样的法证表征 | 是 |
| P3 | 引入 SAM 和检测/分割联合训练 | 将系统扩展到局部篡改定位 | 是，且需要 mask 数据 |

### 9.5 建议增加的实验

两篇论文都通过消融、跨域和退化实验来证明语义—法证融合确实有效。当前项目后续评估至少应包含：

#### 消融实验

- MLLM only；
- MLLM + 单专家；
- 当前三专家 Evidence Token；
- 三专家 + 规则 `EvidenceRectifier`；
- 加入语义 Evidence Token；
- 加入多尺度证据和高亮图；
- Frequency v1 与 v2 对比。

#### 鲁棒性实验

- JPEG 不同质量；
- 图像缩放；
- 高斯模糊与高斯噪声；
- 亮度、对比度和暗化；
- PNG/JPEG 格式配平后的重新评估。

#### 泛化与可信度指标

- 按 ADM、BigGAN、Glide、Midjourney、SD14、SD15、VQDM、Wukong 分模型报告；
- 使用“留一生成器”进行跨生成器测试；
- 除 Accuracy/F1 外，报告 ECE、Brier Score 和 Uncertain 覆盖率；
- 报告证据—结论一致率、区域 IoU、平均专家调用数和单位图像耗时；
- 对专家可靠性按图像格式和退化类型分层统计。

### 9.6 不宜直接照搬的部分

- TVSIP 的 OCR 框校正是文本图像专用设计，一般自然图像应替换为 SAM、边缘或对象候选校正；
- Propose-Rectify 使用 8 张 RTX 4090 进行端到端训练，当前单卡环境更适合先验证规则纠偏和轻量 LoRA；
- 两篇论文主要研究局部篡改，而 GenImage 当前任务多数是整图生成，定位指标的意义需要针对数据类型重新定义；
- 深层特征融合虽然性能更强，但会降低 Evidence Token 的直接可读性。长期设计应同时保留可学习特征和人类可审计的结构化证据；
- 论文结果不能直接作为本项目性能预期，必须在格式配平、跨生成器和独立测试集上重新验证。

总体而言，最值得立即采用的不是直接增加大型分割网络，而是把当前系统从“MLLM 阅读专家分数后自由裁决”升级为“MLLM 提出假设—法证模块显式纠偏—MLLM解释纠偏结果”。这既保持当前状态机和 Evidence Token 的可解释优势，也更符合两篇论文共同验证的可靠取证范式。

## 10. 第二组参考论文：整图 AI 生成检测与解释

本节分析新加入的两篇论文：

1. [Toward Generalizable Forgery Detection and Reasoning](ref/Toward_Generalizable_Forgery_Detection_and_Reasoning.pdf)，IEEE TIP 2026，方法简称 FakeReasoning；
2. [ForenX: Towards Explainable AI-Generated Image Detection with Multimodal Large Language Models](ref/forenx.pdf)，提出 ForenX 与 ForgReason 数据集。

与第 9 节的局部篡改定位论文相比，这两篇论文直接研究“真实照片 vs 完全由生成模型产生的图像”，因此与当前 GenImage 任务更接近。它们最重要的共同结论是：通用 MLLM 的内容理解能力不能自动转化为法证能力；必须给视觉表示加入真实性约束，并同时约束检测结论与自然语言解释。

### 10.1 与当前系统的关键对照

| 维度 | 当前 Forensic-Agent | FakeReasoning | ForenX |
|------|----------------------|---------------|--------|
| 任务 | 整图 Real/Fake/Uncertain，主动调用外部专家 | 整图 Real/Fake + 分层推理 | 整图 Real/Fake + 人类可理解解释 |
| 视觉输入 | Qwen 原始视觉表示 + 专家文本 Token | 冻结 CLIP 与 DINO 双编码器 | CLIP 内容特征 + forensic prompt |
| 法证约束 | 专家 strength 通过文本回灌，无联合检测损失 | FAFF 融合 + 分类概率映射损失 | forensic projector + 辅助检测损失 |
| 结论来源 | 解析模型生成的 verdict JSON | 从固定结论位置的 real/fake token 概率映射 | LLM 输出与辅助 detector 共同训练 |
| 推理结构 | planning / call / reasoning / verdict | summary / caption / hierarchical reasoning / conclusion | 判定 + 分点证据说明 |
| 数据质量 | 原始 577 条 SFT 数据，以自动构造为主；现已自动拒绝 38 条严重伪冲突 | 机器生成后多轮专家审核 | 大规模弱标注预训练 + 2,215 张人工区域解释 |
| 泛化评估 | 尚未系统执行 | train-one-test-many，覆盖 10 类生成器 | 单源训练，跨 Diffusion/GAN 源测试 |
| 当前最值得借鉴 | — | 结论概率映射、结构化推理、CLIP+DINO 互补 | forensic prompt、辅助检测约束、两阶段数据训练 |

### 10.2 FakeReasoning 的可借鉴设计

#### 10.2.1 重新定义“定位”的含义

FakeReasoning 明确指出：对完全生成的图像而言，每个像素理论上都来自生成模型，因此传统篡改掩码或显著性定位不能直接解释“哪些像素被伪造”。这与当前项目非常相关。

当前模型仍会选择 bbox 调用专家，但这些 bbox 应被定义为：

> **诊断证据区域（diagnostic evidence region）**：包含异常纹理、结构、光照、文字或物理关系，能够支持真实性判断的区域；它不是对“篡改像素”的断言。

因此，后续 Trace、UI 和论文描述中应避免把当前 bbox 称为 forged region 或 manipulation mask。若未来混合局部篡改数据，则需要在任务元数据中显式区分 `fully_generated` 与 `locally_manipulated`。

#### 10.2.2 将自由文本 verdict 变成可校准概率

FakeReasoning 的 Classification Probability Mapper（CPM）在固定 `<CONCLUSION>` 位置找到分类 token，并把词表中 `real` 与 `fake` 的 logits 映射为二分类概率，再联合语言建模损失训练。该设计解决了“独立分类头给出一个结果，而解释文本说出另一个结果”的不一致问题。

当前系统完全依赖生成 JSON 中的 `verdict` 和模型自报 `confidence`，这两项都不是经过校准的概率。建议在 `QwenVLClient` 中增加：

```text
固定结论模板
    → 读取 Real / Fake / Uncertain 候选 token logits
    → softmax 得到 model_class_probs
    → 与 JSON verdict 交叉检查
    → 再交给 EvidenceRectifier 融合法证证据
```

建议在 Trace 中增加：

```json
{
  "model_class_probs": {
    "Real": 0.18,
    "Fake": 0.69,
    "Uncertain": 0.13
  },
  "text_verdict": "Fake",
  "verdict_consistent_with_logits": true
}
```

由于当前项目包含 `Uncertain`，不能直接复制论文的二分类映射；应增加第三个候选 token，或根据 Real/Fake 概率间隔及证据冲突规则派生 Uncertain。

#### 10.2.3 使用分层、可审计的推理 Schema

FakeReasoning 将输出分为 `<SUMMARY>`、`<CAPTION>`、`<REASONING>` 和 `<CONCLUSION>`，其中 reasoning 又区分低层视觉痕迹与高层语义异常。当前协议已经有 planning/reasoning/verdict，但视觉观察、专家事实和因果结论仍容易混在同一段文本中。

可将 SFT 目标改为以下结构：

```xml
<summary>说明分析目标与当前证据状态</summary>
<observation>只描述图像中可直接核验的现象</observation>
<forensic_evidence>逐条引用 Evidence ID、区域、可靠性和实际测量值</forensic_evidence>
<reasoning>区分支持、反证、替代解释和剩余不确定性</reasoning>
<verdict>{...}</verdict>
```

这里的重点不是要求模型暴露冗长思维过程，而是让最终审计记录明确区分“看见的事实”“工具测量”“解释”和“结论”。

#### 10.2.4 引入语义与局部表征互补

FakeReasoning 使用冻结的 CLIP 和 DINO ViT-L/14：CLIP 提供全局语义表示，DINO 提供更细粒度的局部视觉信息。FAFF 让 CLIP token 作为 query、DINO token 作为 key/value，并使用 DINO 高层注意力图作为交叉注意力偏置。论文消融表明该融合方向优于反向融合，高层 DINO 注意力也更有效。

对当前项目有两层启发：

- **短期**：保留外部专家，但在 Evidence Token 中分开 `semantic_cue` 与 `low_level_cue`，并让 MLLM 的场景假设决定专家适用度；
- **中长期**：增加轻量 DINO/CLIP 特征分支，把局部视觉特征作为可学习的 forensic token 注入 Qwen，而不是只注入三个标量的自然语言描述。

该方向需要 GPU 训练，不应阻塞当前 539 条候选 SFT 数据及 38 条拒绝记录的格式和质量修正。

#### 10.2.5 数据审核流程比扩大数量更优先

FakeReasoning 的 MMFR 包含约 12 万张图像和 37.8 万条推理标注，覆盖 10 类生成器。其数据不是直接接受模型生成文本，而是先过滤标签不一致、低置信或格式错误样本，再由法证研究人员执行 accept/reject/unsure 审核；不确定样本进入多人复核。论文报告人工审核后平均每图推理数从 3.75 降至 3.15，约 16% 候选推理被拒绝。

这说明原始 577 条数据在进入 LoRA 前必须先做质量审核，而不是默认“格式正确即内容正确”；目前已自动隔离 38 条严重伪冲突，其余 539 条仍需内容审核。建议最少建立以下审计字段：

- `observation_visible`：图像中是否确实存在所述现象；
- `evidence_supported`：专家 raw metric 是否支持文字解释；
- `causal_overclaim`：是否把相关性写成确定因果；
- `label_consistent`：推理、Evidence Token 与 GT 是否一致；
- `review_status`：accept / reject / unsure；
- `reviewer_count`：是否经过第二人复核。

#### 10.2.6 评测解释时不能只用 BLEU/ROUGE

论文指出开放式法证推理可能存在多个有效答案，参考答案不完备、文本冗余和幻觉都会使传统文本相似度失真。其人工评估还发现模型解释具有一定模板化倾向。当前项目应以人工可核验性为主，LLM-as-a-judge 只能作为辅助，并应报告重复推理的一致率。

### 10.3 ForenX 的可借鉴设计

#### 10.3.1 把法证信号做成独立 Prompt，而非普通文字提示

ForenX 在同一个 CLIP 视觉特征上构造两条输入：content projector 保留图像内容表示，forensics projector 生成专门的 forensic prompt。后者通过辅助检测损失 `L_detection` 获得真实性约束，再映射到 LLM 的词嵌入空间。论文消融中，加入 LLM、forensics projector 和检测损失后，GenImage 平均准确率逐步从 84.7% 提升到 97.8%。

这验证了当前 Evidence Token 思路的方向，但也暴露其不足：当前 Token 只是自然语言上下文，没有被检测目标直接约束。可采用渐进式实现：

1. **当前可做**：为每条 Evidence Token 增加 `reliability`、`expected_signal` 和 `counter_explanation`，由确定性 Rectifier 消费；
2. **LoRA 阶段**：加入专用 `<forensic_evidence>` token，并要求最终标签损失与解释生成共同优化；
3. **后续训练**：将专家 heatmap、CLIP/DINO 或频域特征通过 projector 映射为连续 forensic tokens。

#### 10.3.2 检测与解释采用两阶段训练

ForenX 先在 GenImage 与 ForenSynths 上利用内容描述和检测标签进行第一阶段训练，再使用 ForgReason 做解释对齐。ForgReason 中有 2,215 张 Midjourney 写实图像的人工区域与原因标注；为防止只含假图导致过拟合，第二阶段又加入 5,000 张真实图和 1,000 张假图，共 8,215 张。

对当前项目，推荐的训练顺序应调整为：

```text
阶段 1：格式与检测能力
539 条候选 Trace 清洗 + 38 条拒绝记录复核 + 标签平衡 + 固定输出协议
              ↓
阶段 2：解释与证据对齐
少量人工高质量诊断区域 + 原因 + 反例解释
              ↓
阶段 3：跨生成器与退化校准
留一生成器验证 + JPEG/缩放/噪声/后处理
```

人工标注不需要一开始覆盖全量数据。优先选择当前模型与专家分歧最大、置信度最高但判断错误、以及视觉上写实度最高的 hard cases，更符合 ForenX 的数据筛选思想。

#### 10.3.3 保留区域标注，但让总结模型受原始证据约束

ForenX 的人工流程是：先框出不合理区域并写原因，再把框的位置转换为空间描述，最后由 GPT-4 Vision 汇总。该流程说明少量 bbox 标注可显著提高解释与人类观察的一致性，也与当前专家调用 bbox 自然兼容。

不过，ForenX 的限制部分也展示了错误和模糊解释，例如把方向盘位置误判为异常、重复强调同一现象、或给出无法验证的细节。因此当前项目在自动汇总时应：

- 保留人工原始描述，不让模型汇总覆盖原标签；
- 要求每条生成解释引用一个 `evidence_id` 或 `region_id`；
- 允许输出“未发现可核验异常”，禁止为配合 Fake 标签强行编造；
- 对位置、数量、人体结构和文字内容做单独的事实核验；
- 把“无法从图像确认”作为合法的不确定性说明。

#### 10.3.4 增加提示词扰动测试

ForenX 的补充实验显示，普通微调 LLaVA 对非常轻微的 prompt 改写也可能明显掉点，而加入 forensic prompt 后对部分同义表达更稳定；但当问题改成“是否存在合成伪影”等不同任务表述时，两者都可能明显退化。

因此，当前系统不能只验证一种 System Prompt。建议建立等价提示集，测试：

- “是否由 AI 生成”与“图像是否真实拍摄”；
- 要求先给结论、后给结论和只返回 JSON；
- 中英文提示；
- 带/不带“法证专家”角色描述；
- 要求查找 artifact 与要求判断 provenance 的任务差异。

指标应包括标签一致率、格式成功率、专家调用序列稳定性和平均置信度变化。

### 10.4 两篇论文合并后对架构的直接建议

```mermaid
flowchart LR
    I["原始图像"] --> C["内容/语义表征"]
    I --> F["法证表征<br/>外部专家 + 可学习 forensic tokens"]
    C --> P["MLLM Proposal<br/>观察 + 假设 + 专家路由"]
    F --> R["EvidenceRectifier<br/>可靠性与概率校准"]
    P --> R
    R --> O["受约束的分类概率<br/>Real / Fake / Uncertain"]
    R --> E["证据绑定解释"]
    O --> V["最终 Verdict"]
    E --> V
```

建议把目标架构分成四个明确责任域：

1. **Proposal**：MLLM 描述可核验观察、提出假设并选择专家；
2. **Forensic Representation**：外部专家和后续可学习分支提供真实性特征；
3. **Rectification & Calibration**：融合 token logits、专家可靠性和冲突证据，产生受约束概率；
4. **Explanation**：只根据已经绑定的证据组织报告，不得改变纠偏后的结论。

这比让同一个生成过程同时承担观察、测量、概率判断和报告撰写更容易审计。

### 10.5 更新后的实施优先级

| 优先级 | 改造项 | 论文依据 | 预计成本 |
|--------|--------|----------|----------|
| P0 | 把 bbox 统一定义为诊断证据区域，并更新 Trace 字段命名 | FakeReasoning 的整图任务定义 | 低，CPU |
| P0 | 对 539 条候选 SFT 样本进行 accept/reject/unsure 内容审核，并复核 38 条自动拒绝记录 | MMFR 专家审核与 ForenX 人工标注 | 人工时间 |
| P0 | 固定结论位置，读取三类 token logits，并与 JSON verdict 核对 | FakeReasoning CPM | 中，Qwen 推理 |
| P0 | 引入 EvidenceRectifier 与专家可靠性字段 | 两篇论文的检测约束原则 | 中，CPU |
| P1 | 重构 SFT Schema，分离观察、证据、替代解释和结论 | FakeReasoning 分层推理 | 中 |
| P1 | 建立留一生成器、退化、重复推理和提示词扰动测试 | 两篇论文的泛化/稳健性实验 | 中，GPU 推理 |
| P1 | 人工标注少量 hard cases 的区域与原因 | ForgReason 两阶段训练 | 中，人工时间 |
| P2 | 增加 CLIP/DINO 或专家 heatmap projector，学习连续 forensic tokens | FAFF 与 ForenX forensic prompt | 高，GPU 训练 |
| P3 | 研究 token 概率、法证特征和解释生成的联合训练 | CPM + `L_detection` | 高，GPU 训练 |

### 10.6 更新后的评测协议

除第 9.5 节已有实验外，建议补充：

- **生成器隔离**：按生成器划分训练/测试，禁止同一来源随机混入两侧；采用 train-one-test-many 或 leave-one-generator-out；
- **新生成器**：除 GenImage 八类外，条件允许时增加 FLUX、SD3、DALL-E 3 等未见来源；
- **后处理鲁棒性**：JPEG quality、缩放、高斯噪声，并加入锐化、对比度、模糊和截图重编码；
- **结论一致性**：比较 token logits、JSON verdict、报告措辞和 Rectifier 结果是否一致；
- **重复稳定性**：同一样本多次推理，报告标签翻转率、调用序列变化和证据引用变化；
- **解释质量**：人工评估准确性、相关性、合理性、完整性和可核验性，且对评审隐藏方法名称；
- **错误分层**：分别统计高质量假图、非常规真实结构、长焦/运动模糊和后处理真实图；
- **系统成本**：平均调用轮数、各专家调用率、单位图像时延和显存占用。

### 10.7 不应直接照搬的部分

- FakeReasoning 使用 13B LLaVA、双 ViT-L/14 和 8 张 A800；当前环境应先验证概率映射与数据审核，不能把其计算配置当作近期实现前提；
- ForenX 的辅助 detector 与 LLM 共同训练，而当前三个专家是不可微的外部程序，不能直接套用同一个损失函数；短期应通过 Rectifier 和 SFT 约束连接两者；
- 两篇论文均以 Real/Fake 二分类为主，当前 `Uncertain` 应保留，并通过概率间隔、证据冲突与覆盖率单独校准；
- 论文中的高准确率不能直接作为本项目预期，数据格式、训练源和测试协议不同，尤其要先排除 GenImage 的 JPEG/PNG 捷径；
- ForenX 的解释仍存在幻觉、模糊和错误事实，说明“给模型 forensic prompt”不能替代证据绑定与人工审计；
- FakeReasoning 的推理文本可能受语言分布支配，视觉模块提升检测不一定同步提升文本指标，因此训练与评测必须分别报告检测和解释结果。

总体判断：**两篇论文都有明显借鉴价值，而且能够进一步收敛当前项目路线。** 近期最值得实施的不是立即叠加更大的视觉编码器，而是先完成三件事：审核现有 SFT reasoning、把 verdict 变成可校准且可交叉核验的概率、让每条解释绑定真实证据。完成这三项后，再投入 CLIP/DINO 或连续 forensic token 的学习式融合，收益与风险会更容易测量。

## 11. ForgeryVCR 对照与后续工作路线

本节分析 [ForgeryVCR: Visual-Centric Reasoning via Efficient Forensic Tools in MLLMs for Image Forgery Detection and Localization](ref/15.ForgeryVCR%20Visual-Centric%20Reasoning%20via%20Efficient%20Forensic%20Tools%20in%20MLLMs.pdf)。该工作使用 MLLM 主动调用 ELA、FFT、NoisePrint++ 与 Zoom-In，将不可见的低层痕迹转换为可重新输入视觉编码器的图像，再输出图像级真假结论和篡改 bbox，并用 SAM2 将 bbox 细化为像素掩码。

ForgeryVCR 与当前 Forensic-Agent 的控制流高度相似，是目前最直接支持“MLLM 主动调度外部法证工具”路线的参考工作。不过，两者的当前任务不同：本项目目前主要检测完全生成图像，ForgeryVCR 主要检测局部拼接、复制移动和移除/修补。因此，后续设计应保持同一套状态机和 Expert 接口，同时对全局生成与局部篡改使用不同的证据语义和输出目标。

### 11.1 当前全局任务与未来局部任务的统一边界

| 维度 | 当前：完全生成图像检测 | 后续：局部篡改检测与定位 |
|------|------------------------|----------------------------|
| 任务类型 | `fully_generated` | `locally_manipulated` |
| 最终标签 | Real / Fake / Uncertain | Real / Manipulated / Uncertain |
| 区域含义 | 诊断证据区域，不表示该区域才是伪造像素 | 候选篡改区域，可与真实 mask 比较 |
| 主要监督 | 图像级来源标签、生成器类型 | 图像级标签 + bbox/mask + 篡改类型 |
| 工具目标 | 判断整图生成统计特征、语义/物理异常 | 找到局部来源、压缩、噪声和边界不一致 |
| 定位指标 | Evidence coverage、区域稳定性，不使用篡改 IoU | BBox-IoU、Pixel F1、Mask IoU |
| SAM2 作用 | 不需要 | bbox 到像素 mask 的可选细化器 |

建议从现在开始在 Trace 中加入：

```json
{
  "task_type": "fully_generated",
  "evidence_scope": "global",
  "region_semantics": "diagnostic_evidence_region"
}
```

未来局部数据则使用：

```json
{
  "task_type": "locally_manipulated",
  "evidence_scope": "local",
  "region_semantics": "candidate_manipulation_region"
}
```

这样可以避免当前 bbox 在训练中被错误理解为篡改真值，也能在后续加入 SAM2 或分割头时保持数据兼容。

### 11.2 ForgeryVCR 对当前架构的关键证据

ForgeryVCR 的核心不是简单增加工具，而是改变工具证据进入 MLLM 的方式：

```text
当前项目：Expert → strength / explanation → Evidence Token 文本 → Qwen
ForgeryVCR：Tool → ELA/FFT/NPP 可视化图 → 视觉编码器 → MLLM
建议方案：Expert → Evidence Bundle（数值 + 可视化图 + 可靠性）→ Qwen + Rectifier
```

其消融结果显示：无 CoT 时检测 F1/ACC 为 0.7351/0.7561；仅增加文本 CoT 后下降到 0.6431/0.7215；视觉与文本并用为 0.7756/0.7711；仅使用视觉中心推理达到 0.8271/0.8261。像素定位 IoU 也从无 CoT 的 0.4357 提升到视觉中心推理的 0.5306。该结果说明低层法证痕迹如果先被压缩成语言，可能出现信息损失和语义幻觉。

因此，Evidence Token 仍应保留用于日志、规则融合和人工审计，但不应继续作为专家证据进入 Qwen 的唯一通道。

论文还给出三项对当前项目特别重要的结论：

1. **工具需要准入测试**：论文从八种候选法证描述子中只保留 ELA、FFT、NPP 和 Zoom-In；继续增加 CFA、DCT、PSCC、VAE residual 与 SRM 后理论性能上限几乎饱和；
2. **轨迹需要按样本增益生成**：只有单工具模型优于无工具基线且超过有效阈值时，该工具才进入该样本的 SFT 轨迹；
3. **SFT 只教会调用格式，RL 才改善选择策略**：论文观察到 SFT 模型倾向机械调用工具，GRPO 后会替换无效工具、删除模糊视图并减少冗余 Zoom-In。

### 11.3 Expert 是否需要更新

结论是：**需要更新，但第一步应更新输出接口和验证方法，而不是立即替换所有算法。**

当前 `ExpertResult` 主要输出 `strength`、`support`、`phenomenon` 与 `reasoning`，随后 `EvidenceTokenizer` 又只保留这些文本和标量字段。该设计存在三个问题：

- 一个标量无法保留异常在 patch 内部的空间分布；
- 不同专家的 strength 没有统一概率含义，不能直接比较或投票；
- Qwen 无法看到 FFT 频谱、噪声残差或 JPEG 块结构，只能相信专家写出的解释。

#### 11.3.1 将 ExpertResult 升级为 Evidence Bundle

建议兼容性扩展如下：

```json
{
  "evidence_id": "E-noise-0003",
  "source": "noise_expert",
  "task_applicability": ["fully_generated", "locally_manipulated"],
  "scope": "global|local",
  "region": [100, 120, 500, 520],
  "region_semantics": "diagnostic_evidence_region",
  "raw_metric": 1.82,
  "calibrated_likelihood": {
    "Real": 0.25,
    "Fake": 0.58,
    "Uncertain": 0.17
  },
  "reliability": 0.63,
  "reliability_factors": {
    "image_format": "PNG",
    "jpeg_quality": null,
    "resolution_sufficient": true
  },
  "visual_artifacts": [
    {
      "type": "noise_residual_map",
      "path": "traces/evidence/E-noise-0003.png"
    }
  ],
  "phenomenon": "...",
  "counter_explanation": "重采样或去噪也可能产生相似现象"
}
```

其中：

- `visual_artifacts` 提供给 Qwen 视觉编码器；
- `raw_metric`、`calibrated_likelihood` 和 `reliability` 提供给 EvidenceRectifier；
- `phenomenon` 与 `counter_explanation` 提供给人工审计和最终报告；
- 旧字段保留一段过渡期，避免立即破坏现有 Trace 与测试。

#### 11.3.2 三个现有 Expert 的具体改造

| Expert | 当前问题 | 全局检测近期改造 | 局部篡改后续改造 |
|--------|----------|------------------|------------------|
| Frequency | v1 分离度弱，patch 标量丢失频谱结构 | 以 v2 为候选，输出全图频谱、径向谱、周期峰和空间频率热图；按生成器与缩放条件校准 | 输出局部/背景频谱差异图和边界上下文特征 |
| Noise | strength 易受 JPEG、去噪、分辨率影响 | 输出噪声残差图、分块方差图、全局一致性分布和条件可靠性 | 接入 NoisePrint++ 或等价模型，比较候选区与背景噪声指纹 |
| JPEG | 当前块效应/DCT 指标可能学习 Real JPEG 与 Fake PNG 捷径 | 增加 ELA 可视化与编码历史估计；在格式配平数据上重新校准 | 比较候选区和背景的压缩历史，生成局部不一致热图 |
| Zoom-In（新增） | 当前裁剪只是状态机内部动作，不是显式工具 | 将高分辨率裁剪注册为 Expert，保留上下文环带和缩放参数 | 支持迭代缩放与 bbox 收缩，作为 SAM2 前的定位步骤 |

不建议现在直接加入全部 ForgeryVCR 工具。应先建立 Expert 准入实验：

```text
Qwen/RGB baseline
RGB + 单 Expert 文本
RGB + 单 Expert 可视化图
RGB + 文本和可视化双通道
```

每个 Expert 至少应报告：

- 全局 Real/Fake 的 Accuracy、F1、AUROC、ECE；
- 按生成器、PNG/JPEG、分辨率和后处理类型分层的增益；
- 相对 RGB baseline 的净提升；
- 平均耗时、失败率和可视化图 token 成本；
- 与其他 Expert 的错误重叠和互补性。

只有稳定产生正增益且与已有工具互补的 Expert 才进入正式工具箱。

### 11.4 停止策略是否需要改进

结论是：**需要，而且应在 Expert 校准后改。** 当前四个条件 `verdict → max_steps → conflict → strength delta` 是固定优先级；ForgeryVCR 则通过 no-tool、single-tool、multi-tool 轨迹和工具效用优化，使模型学习何时直接回答、何时继续调用工具。

论文的样本级工具筛选为：

```text
工具性能 Pt > max(无工具性能 Pbase, 有效阈值 τ)
```

这可以转换为当前项目的停止原则：

> 只有下一工具的预期净增益为正时继续；否则根据融合后的证据输出 Real、Fake 或 Uncertain。

#### 11.4.1 当前实现需要修正的语义

- `<verdict>` 只能视为候选结论，不能绕过证据冲突和概率校准立即退出；
- `MAX_STEPS` 只表示不能继续获取证据，不应直接决定最终标签；
- Evidence conflict 是需要消歧的状态，不一定立即停止；有可靠且未调用的正交专家时应继续验证；
- 当前所谓信息增益实际是最后两个不同专家 strength 的绝对差，并非 KL divergence；不同专家的标量也不具可比性；
- 当前 `step` 统计包含工具调用的模型轮数，一轮可能执行多个专家，不能准确表达调用预算。

#### 11.4.2 建议的决策状态

```json
{
  "posterior": {"Real": 0.24, "Fake": 0.58, "Uncertain": 0.18},
  "candidate_verdict": "Fake",
  "conflict_score": 0.31,
  "expert_call_count": 2,
  "accumulated_cost": 1.7,
  "remaining_tools": ["jpeg_expert"],
  "best_next_action": "jpeg_expert",
  "expected_tool_gain": 0.08,
  "expected_net_utility": 0.05
}
```

#### 11.4.3 建议的停止流程

```mermaid
flowchart TD
    A["MLLM 候选 verdict + token 概率"] --> B["EvidenceRectifier<br/>融合可靠性加权证据"]
    E["Evidence Bundles"] --> B
    B --> C{"存在强冲突？"}
    C -->|是| D{"有预算且存在<br/>正收益消歧工具？"}
    D -->|是| T["调用最有价值的正交工具"]
    D -->|否| U["停止：Uncertain"]
    C -->|否| F{"立即决策风险<br/>是否可接受？"}
    F -->|是| V["停止：Real 或 Fake"]
    F -->|否| G{"最佳工具净收益 > 0<br/>且预算可用？"}
    G -->|是| T
    G -->|否且低置信| U
    G -->|否且置信足够| V
```

建议让 `HaltingChecker` 返回决策对象而不是 `(bool, reason)`：

```text
HaltingDecision
├── action: continue / halt
├── verdict: Real / Fake / Uncertain / null
├── primary_reason
├── all_reasons
├── next_expert
├── posterior
├── conflict_score
└── expected_net_utility
```

停止原因应允许同时存在。例如达到预算且证据冲突时：

```json
{
  "primary_reason": "unresolved_evidence_conflict",
  "all_reasons": ["max_budget_reached", "evidence_conflict"],
  "verdict": "Uncertain"
}
```

#### 11.4.4 工具增益的实现顺序

短期尚无学习式增益模型时，可使用校准集统计构造：

```text
expected_gain(tool)
  = historical_accuracy_gain
  × condition_applicability
  × current_uncertainty
  × non_redundancy
  × reliability

expected_net_utility
  = expected_gain - call_cost - latency_cost
```

中期收集足够 Trace 后，再训练轻量路由器预测：

```text
当前后验 + 图像元数据 + 已调用工具 + 工具摘要
    → 每个候选工具的预期风险下降
```

不应直接复制 ForgeryVCR 的工具奖励公式。其公开 `Rtool` 主要依据“是否调用工具且最终正确/IoU 超阈值”给正奖励，没有显式按调用次数扣分，一个工具与多个工具可能得到相同工具奖励。当前项目更适合：

```text
Rtool
  = 任务风险下降
  - λcall × 调用次数
  - λrepeat × 重复/同源调用
  - λlatency × 运行成本
  - λinvalid × 无效或不适用调用
```

### 11.5 SFT 数据是否需要人工审计

结论是：**需要，并且应在任何 LoRA 训练之前完成。** 原始 577 条数据已完成第一轮冲突结构审计，当前保留 539 条候选训练数据，并将 38 条严重伪冲突移入拒绝集。其中：

- `correct=196` 只保证最终 verdict 与 GT 一致，不保证观察、专家引用和因果解释正确；
- 原 `conflict=181` 中已有 38 条因证据同向或重复 Expert 被自动拒绝；保留的 `conflict=143` 与 `borderline=100` 仍为合成数据，需要继续检查模板捷径和真实专家行为；
- `format=100` 的元数据已明确说明内容可能错误，只能用于格式训练，不能默认用于事实性 reasoning 训练；
- 旧 Trace 主要依赖文字 Evidence Token，尚未包含 ForgeryVCR 式可视化证据。

当前 539 条候选数据仍适合进行全量人工首审。建议采用“单人全量 + hard case 双人复核”，并保留 38 条拒绝记录用于追踪生成规则缺陷。

#### 11.5.1 审计字段

每条数据至少记录：

```json
{
  "sample_id": "...",
  "review_status": "accept|revise|reject|unsure",
  "image_label_correct": true,
  "visible_observation_grounded": true,
  "expert_value_matches_text": true,
  "expert_applicable_to_input": true,
  "reasoning_overclaims_causality": false,
  "verdict_supported_by_evidence": true,
  "bbox_semantics_correct": true,
  "format_valid": true,
  "suspected_shortcut": null,
  "reviewer_notes": "...",
  "second_review_required": false
}
```

重点检查以下失败模式：

- 图中不存在 reasoning 描述的视觉异常；
- strength/support 与 reasoning 相互矛盾；
- 把 PNG/JPEG 格式差异写成生成证据；
- 把相关性描述成确定因果；
- Fake 标签导致模型必须编造一个可疑区域；
- conflict 样本只是人为拼接高低 strength，没有真实物理冲突；
- format 样本虽然 XML/JSON 合法，但内容结论错误；
- bbox 被描述成篡改区域，而当前数据其实是整图生成。

#### 11.5.2 审计后的数据分层

不要把所有样本混合成一个训练集，建议形成：

| 层级 | 用途 | 内容要求 |
|------|------|----------|
| A：事实金标 | detection + evidence alignment | 人工确认图像观察、专家值和结论一致 |
| B：工具策略 | tool routing | 包含 no-tool、single-tool、multi-tool 的真实增益轨迹 |
| C：冲突与拒判 | Uncertain calibration | 真实可靠专家冲突、低置信和工具失效样本 |
| D：格式恢复 | syntax repair | 只训练标签/XML/JSON 修复，不参与法证事实学习 |
| R：拒绝集 | 不训练 | 幻觉、标签污染、无法修正或证据不可核验 |

#### 11.5.3 借鉴 ForgeryVCR 重建增益轨迹

在完成 Expert 单工具基线后，为每张图生成：

```text
[]                              no-tool
[best_tool]                     single-tool
[second_independent_tool]       alternative single-tool
[best_tool, second_tool]        multi-tool verification
```

但只保留实际改善图像级分类风险或证据可靠性的路径。不能仅因为某工具生成了图就认为它有信息增益。

对当前全局任务，性能 `P` 应优先使用校准后的图像级分类概率、负对数损失或 Brier Score，而不是局部 IoU。进入局部篡改阶段后，再把 bbox/mask IoU 加入工具收益。

### 11.6 全局优先、局部兼容的实施路线

```mermaid
flowchart LR
    G0["G0 数据审计<br/>539 条候选全量首审"] --> G1["G1 Expert 基线<br/>格式配平 + 单工具增益"]
    G1 --> G2["G2 视觉证据<br/>Evidence Bundle + 多图回灌"]
    G2 --> G3["G3 决策停止<br/>Rectifier + 工具净收益"]
    G3 --> G4["G4 增益 SFT<br/>no/single/multi-tool"]
    G4 --> G5["G5 可选 GRPO<br/>准确率 - 成本 - 冗余"]
    G5 --> L1["L1 局部数据<br/>bbox/mask + 篡改类型"]
    L1 --> L2["L2 定位工具<br/>局部热图 + Zoom-In"]
    L2 --> L3["L3 精细分割<br/>SAM2 或分割头"]
```

#### G0：现有数据审计

- 全量审计 539 条候选 SFT 数据，并复核 38 条拒绝记录的规则命中情况；
- 分离事实训练、工具策略、冲突拒判和格式修复数据；
- 暂停直接使用未经审计的 reasoning 进行 LoRA。

#### G1：Expert 准入与校准

- 建立格式配平的 Real/Fake 验证集；
- 分别测试 Frequency v1/v2、Noise、JPEG/ELA 和 Zoom-In；
- 保存每个样本的原始指标、预测、可视化图和耗时；
- 删除无稳定增益或高度冗余的 Expert。

#### G2：视觉证据通道

- 扩展 `ExpertResult` 与 Trace Schema；
- 让专家落盘可视化证据图；
- 修复 Qwen 多轮消息，使原图和工具图可共同进入后续轮次；
- 对比“文本 only / 视觉 only / 双通道”三组消融。

#### G3：停止策略与校准

- 提取 Real/Fake/Uncertain 候选概率；
- 实现 EvidenceRectifier 和条件可靠性；
- 用 `expert_call_count` 与加权成本替代当前轮数预算；
- 用后验变化和候选工具净收益替代 strength delta。

#### G4：增益驱动 SFT

- 按真实工具增益构造 no-tool、single-tool 和 multi-tool 轨迹；
- 保留简单样本直接回答能力；
- 让复杂样本学习正交验证而不是固定工具顺序；
- 单独评估工具选择准确率、无效调用率和平均调用数。

#### G5：可选 GRPO

只有 SFT 后能够稳定输出合法调用、Expert 已完成校准且 reward 可离线复算时才进入。否则 RL 会放大格式捷径、数据格式偏差或虚假的专家指标。

#### L1-L3：局部篡改扩展

- 引入带 bbox/mask 的 CASIA、IMD2020、Coverage 等局部篡改数据；
- 增加 `task_type`，避免整图生成与局部篡改监督混淆；
- 训练 MLLM 输出候选篡改 bbox；
- 评估原始 BBox-IoU 后再接 SAM2，避免用分割器掩盖粗定位不足；
- 联合报告图像级检测、bbox 定位、像素分割和工具成本。

### 11.7 阶段门槛与下一步优先级

| 顺序 | 工作项 | 完成门槛 | 是否需要 GPU |
|------|--------|----------|--------------|
| 1 | 539 条候选 SFT 人工审计 | 100% 有审核状态；38 条拒绝记录可追溯；unsure 完成二审 | 否 |
| 2 | 格式配平与单 Expert 基线 | 每个工具有分层增益、可靠性和成本报告 | Qwen 对比需要 |
| 3 | Evidence Bundle 与可视化图 | 三个 Expert 均能生成可审计 artifact，旧接口兼容 | 否 |
| 4 | Qwen 多图回灌 | 原图和工具图在多轮中不丢失，消融可复现 | 是 |
| 5 | 停止策略 v2 | 无效调用率下降；冲突时不被候选 verdict 覆盖 | CPU 可测，校准需 GPU |
| 6 | 增益驱动 SFT | no-tool 能力保留，单/多工具调用与真实增益一致 | 是 |
| 7 | 局部篡改数据与定位 | BBox-IoU 独立达标后再接 SAM2 | 是 |

近期优先级建议为：

1. **先人工审计 SFT 数据**：这是避免把错误 reasoning 和格式捷径写入 LoRA 的最低成本措施；
2. **再建立单 Expert 增益基线**：没有该基线就无法定义工具效用和停止条件；
3. **随后升级 Expert 为数值 + 视觉双通道**：先保留现有算法，验证可视化证据是否真正帮助 Qwen；
4. **最后改停止策略并重建 SFT 轨迹**：停止逻辑依赖校准后的后验、可靠性和工具收益，过早修改只能换一组启发式规则；
5. **全局检测稳定后再进入局部定位**：复用相同工具协议，但新增 bbox/mask 监督和 SAM2，不混淆两类任务的区域语义。

总体判断：Expert、停止策略和 SFT 数据都需要调整，但它们存在明确依赖关系。**SFT 审计提供可信监督，Expert 校准提供工具真实增益，二者共同决定停止策略；停止策略稳定后，才适合生成新的工具轨迹并开展 SFT/GRPO。**
