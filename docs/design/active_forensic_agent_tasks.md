# 任务书：基于主动探索型 MLLM 与辅助法证专家组的图像真实性检测系统（原型开发阶段）

## 1. 项目背景与技术定位

本项目旨在构建一个**"主动探索型（Active Exploration）双分支图像取证系统"**。

*   **传统方案的痛点**：多模态大语言模型（MLLM）长于高层语义理解，但受限于其视觉编码器（如 CLIP 等）的降采样和特征选择，对频域网格、像素噪声分布、JPEG 压缩残差等微观物理痕迹处于"睁眼瞎"状态；而传统的深度检测模型（分类器）抗噪能力弱，且无法给出让人类信服的可解释性法证报告。
*   **我们的核心创新（Our Edge）**：MLLM 作为高层"法官"，通过主动发出动作 Token，在图像的可疑局部区域（RoI）上动态、按需地调用底层"物理法证专家组"；专家组的物理指标被抽象为包含物理现象、原理解释和置信度的结构化 `Evidence Token` 反馈给 MLLM。MLLM 结合宏观语义上下文（常识、物理规律等）对证据进行交叉质证，最终生成一份逻辑链条不可推翻的、可解释性的真伪判定报告。

---

## 2. 核心架构设计

系统数据流与控制流采用**"路线A：纯 Python 状态机循环"**，结构解耦，不依赖 LangChain 等重型 Agent 框架，保证代码的极致干净与可控性。

```
                    [输入图像 Image]
                          │
        ┌─────────────────┴─────────────────┐
        ▼                                   ▼
  [MLLM 主控分支] ◄────────── [状态机控制器 (State Loop)]
  (语义分析与决策)                  │
        │                           │ 2. 按需调用 (Action Token)
        │ 1. 结构化 SOP 输出        ▼
        │   (包含 <call_*>[bbox]) ──► [法证专家组 (Forensic Experts)]
        │                           (局部切片微观信号分析)
        │                                   │
        │                                   │ 3. 物理信号分析
        ▼                                   ▼
  [生成可信法证报告] ◄───────────────── [Evidence Token 抽象层]
                                      (转换为 Discretized Schema)
```

---

## 3. 跑通 3 个经典法证专家模块

你需要在 `experts/` 目录下，使用 PyTorch 和 OpenCV/SciPy 实现并封装以下 3 个经典的局部图像法证专家模块（入参统一为：`img_np: np.ndarray`, `bbox: list[int]`）。

### 3.1 频域指纹专家 (Frequency Expert - `freq`)

*   **物理依据**：GAN/Diffusion 等生成模型在反卷积或上采样时，极易在局部高频波段留下周期性的网格伪迹（Grid Artifacts）。
*   **算法实现**：对裁剪出的 `bbox` 局部图像进行双三次二维傅里叶变换（2D-FFT），计算高频功率谱，检测是否有异常的周期性高频尖峰。
*   **输出强度 $s$**：归一化异常值 $s \in [0, 1]$。

### 3.2 噪声残差专家 (Noise Expert - `noise`)

*   **物理依据**：真实相机成像在局部有高度一致的微观传感器噪声（PRNU）或自然的噪声方差分布。而局部拼接、AI 局部重绘（Inpainting）会破坏边缘处的噪声连续性，导致残差不一致。
*   **算法实现**：利用经典高通滤波器（如 SRM 空间富模型滤波器组中的单个核心滤波器）提取 `bbox` 区域的噪声残差图，并计算局部方差。若方差与背景整体出现明显断层，则断层程度越明显异常度越高。
*   **输出强度 $s$**：归一化异常值 $s \in [0, 1]$。

### 3.3 JPEG 压缩专家 (JPEG Expert - `jpeg`)

*   **物理依据**：图像在传输或被二次编辑保存时，会留下一组特定的量化表特征。局部拼接（Splicing）和不匹配的压缩会导致 DCT 系数直方图出现不自然的"双重 JPEG 压缩"网格量化断层。
*   **算法实现**：计算该 `bbox` 局部区域的 JPEG 块效应伪迹强度（Blockiness Artifacts）或量化误差图。
*   **输出强度 $s$**：归一化异常值 $s \in [0, 1]$。

---

## 4. 关键接口定义：数据抽象层（Evidence Tokenizer）

专家模块的标量输出（如 `0.85`）对 MLLM 缺乏物理概念。为防止大模型沦为只看数值的计算器，抽象层必须将数字硬分转化为**"语义软描述"**再注入上下文。

### 4.1 Evidence Token Schema (JSON 格式定义)

```json
{
  "evidence_name": "abnormal_high_frequency_residual",
  "region": "patch_coordinates_[y_min, x_min, y_max, x_max]",
  "phenomenon": "The spatial-frequency spectrum of this localized boundary shows unusually concentrated grids.",
  "reasoning": "This spectral pattern is mathematically consistent with upsampling/deconvolution grid artifacts common in artificial image synthesis.",
  "strength": 0.82,
  "source": "frequency_expert",
  "support": "AI-generated",
  "interpretation_text": "Severe statistical anomaly matching artificial generative fingerprints."
}
```

### 4.2 强度映射字典（Discretized Text Mapping）

*   $0.0 \le s < 0.3$ → "Statistical patterns align with normal hardware camera capture."
*   $0.3 \le s < 0.7$ → "Mild mathematical distortions noted; localized compression or blurring suspected."
*   $0.7 \le s \le 1.0$ → "Severe statistical anomaly matching artificial generative fingerprints."

---

## 5. 状态机控制与 SOP 调用决策树

### 5.1 智能体 System Prompt 强制 SOP 约束

为了防止模型"胡乱调用"和"盲目决策"，必须要求 Qwen2.5 在生成工具调用指令前，强制在 `<planning>` 标签内完成一个"结构化诊断病历"：

*   **Suspected Region（我锁定的可疑区域）**：提供具体 BBox 坐标（相对归一化，格式 `[y_min, x_min, y_max, x_max]`）。
*   **Visual Anomalies（我观察到的视觉异常）**：必须从"边缘不自然/纹理过平滑/重压缩马赛克/物理光影冲突"中选择至少一项。
*   **Expert Target & Hypothesis（拟调用的专家及物理依据）**：说明为什么要调用该专家以及你预期的物理假说。

**调用规则约束字典**：

*   针对怀疑有拼接、虚化或异常锐化的边缘：必须优先调用 `noise` / `freq` 专家；
*   针对大面积、过度平滑或过度规则的生成纹理：必须优先调用 `freq` 专家；
*   针对低像素、存在方块效应或疑似经过社交媒体重压缩的图像：必须在首轮先调用 `jpeg` 专家。

### 5.2 Python 状态机多重拦截与终止机制 (Halting Criteria)

状态机在将数据喂给 MLLM 前，会通过 Python 后台执行双轨制拦截，控制循环最大步数限制在 $5$ 步以内。

#### 轨道一：模型主动结案

若模型认为已有证据（$\Delta_1, \Delta_2, \dots$）已在物理和语义层构成闭环，大模型输出 `<verdict>` 标签，状态机主动结束循环。

#### 轨道二：状态机硬性条件拦截 (Safety Guards)

*   **拦截 A：最大步数硬封顶**：当执行完第 5 轮工具调用后，即使模型未输出 `<verdict>`，状态机也将强制拦截并追加提示："取证资源（Budget）已耗尽，请立即基于已获取的全部证据撰写最终报告并输出判断。"
*   **拦截 B：信息增益收敛 (Information Gain)**：若在连续两轮调用后，新反馈的 Evidence Token 提供的置信度变化极低（如标量分极小，判定为 Neutral），表明无新线索，状态机强制终止调用。
*   **拦截 C：证据强冲突（疑罪从无 - Conflict Halting）**：当频域专家强烈判假（AI-generated, strength: 0.9）、但噪声与压缩专家强烈判真（Real, strength: 0.95）时，说明底层物理痕迹出现强冲突。状态机检测到此情况后立即中止循环，要求模型进行双向反思，并输出 Uncertain 置信度校准结论。

---

## 6. 防"数值懒惰"的证据融合与可解释报告生成

在输出最终判定（`<verdict>`）前，必须约束 MLLM 在其 `<reasoning>` 标签内执行"双重证据质询"，绝不能只复述标量分数（Strength）：

*   **环境污染质询**：模型必须分析图像后处理对物理专家的干扰。例如：若图像存在重 JPEG 压缩痕迹，即使 freq 专家的异常度只有 0.4，模型也需要结合语义说明"因压缩污染导致物理指纹变弱，不能单凭低分排除嫌疑"。
*   **物理-语义一致性校验**：模型必须阐述专家模型提取的物理现象（phenomenon）和它在视觉层观察到的瑕疵（如猫咪边缘有锯齿、手部异常等）是否能在科学因果链上完全吻合。

---
## 7. 核心模型对齐基建：SFT（监督微调）数据工程与训练规范

由于基座模型（如 Qwen2.5-VL / Llama3.2-Vision）缺乏本系统特有的状态机 SOP 标签和工具调用意识，必须通过两阶段微调（SFT）来激活并固化其 Agent 行为特征。

### 7.1 SFT 核心能力对齐目标
1. **格式硬收敛**：使模型形成对 `<planning>`, `<call_*>`, `<reasoning>`, `<verdict>` 标签的绝对条件反射，输出错漏或不闭合率低于 0.1%。
2. **多轮 Session 状态感应**：使模型能精准分辨“首轮观察图像、中间轮次接收物理 Evidence Token、尾轮总结结案”的不同阶段职责。

### 7.2 SFT 训练数据 Schema 设计 (ShareGPT 格式单轮/多轮示例)
数据构造包含两条主线：**A. 正常破案流数据**（1~3 轮工具调用后成功结案）；**B. 拦截与冲突对抗流数据**（状态机硬性拦截后，迫使模型进行双向反思和置信度校准）。

```json
{
  "id": "forensic_sft_session_042",
  "conversations": [
    {
      "from": "user",
      "value": "<image>\n请分析这张图像的真实性，并使用法证工具箱开展多轮质证。"
    },
    {
      "from": "gpt",
      "value": "<planning>\nSuspected Region: [210, 150, 480, 420]\nVisual Anomalies: 图像中央人脸边缘存在不自然的羽化与模糊，光影与背景有轻微冲突。\nExpert Target & Hypothesis: 拟调用 noise 专家。边缘重绘或拼接会破坏微观传感器噪声（PRNU）的一致性，预计残差方差会出现明显断层。\n</planning>\n<call_noise>[210, 150, 480, 420]</call_noise>"
    },
    {
      "from": "user",
      "value": "{\"evidence_name\": \"noise_residual_inconsistency\", \"region\": \"patch_coordinates_[210, 150, 480, 420]\", \"phenomenon\": \"Localized PRNU sensor noise variance measures abnormally at 3.12.\", \"reasoning\": \"The micro-noise pattern exhibits localized variance collapse...\", \"strength\": 0.85, \"source\": \"noise_expert\", \"support\": \"Tampered/Spliced\"}"
    },
    {
      "from": "gpt",
      "value": "<reasoning>\n【物理-语义一致性校验】底层 noise 专家反馈该区域噪声方差发生严重塌陷（仅为 3.12），这与我在视觉上观察到的边缘过分平滑、羽化瑕疵在因果链上完全吻合。这证明该区域不是相机硬件直接成像，而是经过了后期的 AI 局部擦除重绘（Inpainting）或软件平滑羽化。\n</reasoning>\n<verdict>\n{\"verdict\": \"Fake\", \"confidence\": 0.92, \"primary_evidence\": \"noise_residual_inconsistency\", \"report\": \"图像人脸区域经过后期伪造。微观噪声残差分析证实该局部存在严重的方差塌陷，物理指纹断层与视觉边缘平滑瑕疵高度一致。\"}\n</verdict>"
    }
  ]
}
```
---

## 8. 高级功能预留：GRPO（Group Relative Policy Optimization）强化学习对齐接口

为后续的学术训练和对齐（Post-training Alignment）做准备，系统需在状态机日志和数据保存模块中，设计并保存每一轮交互的 Trace Log[cite: 1]。未来使用 GRPO 进行强化对齐时，将使用以下 Rule-governed Reward（规则奖励）进行组内相对打分：

*   **Format Reward** ($+0.2$ / $-0.5$)：是否严格遵循 SOP 的 `<planning>`, `<call_*>`, `<reasoning>`, `<verdict>` 格式[cite: 1]。
*   **Anti-Numerical Laziness Reward** ($+0.5$ / $-0.6$)：模型在最终分析中是否将专家反馈的定性核心词（如 grid residual）与视觉瑕疵进行交叉绑定，而非仅仅提及"分数"[cite: 1]。
*   **Attention-Evidence Consistency Reward** ($+0.4$)：最终判定报告声称有异常的区域，与调用专家时使用的 BBox 是否具有空间一致性（高 IoU）[cite: 1]。
*   **Outcome Accuracy Reward** ($+1.0$ / $-1.0$)：分类正确性，并对在极端冲突下机智地判定为 Uncertain 的表现给予额外奖励（$+0.5$）[cite: 1]。

---

## 9. Claude Code 执行指南（面向 Claude Code）

请按照以下模块化步骤，开始生成项目的初始代码结构：

### 阶段一：基础架构与 Mock 专家组

*   在项目根目录下，创建 `mainy`.p（主状态机核心循环）。在根目录下生成 `requirements.txt`。
*   创建 `experts/` 文件夹，在其中新建 `frequency.py`, `noise.py`, `jpeg.py`，实现一个统一的 `BaseExpert` 类。
*   在第一步中，物理计算逻辑可以用简易的 Mock 逻辑代替（读取图像后根据固定的图像哈希或模拟随机数输出 strength、phenomenon、reasoning 等 JSON 信息，以快速跑通数据管道）[cite: 1]。
*   实现 `utils/coordinate_transformer.py`，负责在 MLLM 输入相对坐标与 Python 原生 OpenCV 绝对像素坐标之间建立转换。
*   大模型输出的 bbox 为相对归一化坐标 [ymin, xmin, ymax, xmax]，范围为 $0 \sim 1000$（对齐 Qwen2.5-VL 规范）。coordinate_transformer.py 必须将其转换为 OpenCV 的绝对像素坐标 [ymin_px, xmin_px, ymax_px, xmax_px]。

### 阶段二：解析器与容错反馈机制

*   在 `utils/parser.py` 中编写强大的正则表达式解析器，专门抓取 `<planning>`、`<call_expert_name>[y_min, x_min, y_max, x_max]</call_expert_name>` 等标签。
*   实现 "格式纠错并喂回（Feedback Loop）" 机制：若解析失败，在不崩溃的前提下向大模型发送一条带有错误警告的隐式 User 提示，要求重构工具调用。

### 阶段三：真实专家算法替换

完善 `experts/` 下的算法实现，引入 OpenCV、PyTorch、SciPy 的 2D-FFT、高通滤波器以及 JPEG 量化表检测逻辑，替代 Mock 逻辑。
