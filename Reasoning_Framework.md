# 多层法证证据驱动的 MLLM 图像真实性检测框架与主动取证系统设计

> **文档定位（2026-09-20）**：本文保留项目早期研究构想与理论动机，其中熵/KL 停止、注意力对齐损失等内容是概念设计，不代表当前代码已经实现。当前运行逻辑见 `CURRENT_PROGRAM_ARCHITECTURE.md`，后续唯一执行计划见 `plan.md`。

---

## 一、 总体结论与学术定位

在 AIGC 检测和数字图像取证领域，当前的学术趋势正在从“二分类准确率”走向“可解释、可验证、可泛化的证据驱动判断”。然而，主流的多模态大语言模型（MLLM）在处理此类任务时存在致命的“盲区”：它们主要依赖视觉编码器（如 CLIP）提取高层语义特征，这使得模型对频域特征、微观噪声和数字物理痕迹等“非语义”底层特征几乎处于“睁眼瞎”的状态。

为了攻克这一瓶颈，我们提出了**基于辅助法证分支的主动探索型 MLLM 图像真实性检测框架**。该框架将 MLLM 的高层逻辑推理、常识判断能力，与传统法证专家模型（CNN/传统机器学习）的底层物理特征提取能力相结合，实现“机器证据驱动的可信法证报告”。

### 1.1 前沿工作（ICLR 2026）与我们思路的关联分析
我们可以用一句话理清当前最前沿的工作与我们框架的学术定位关系：

> **FakeXplain** 教模型“看哪里假”；**VERITAS** 教模型“如何一步步推理”；**AnomReason** 教模型“如何描述语义不合理”；**X-AIGD** 规定了“哪些伪迹该被定位”。
> 
> **我们的核心差异（Our Edge）：** 
> *FakeXplain teaches MLLMs to reason over human-annotated visible artifacts, while our method aims to teach MLLMs to reason over machine-extracted forensic evidence.* 我们不只让 MLLM 局限于人类肉眼可见的瑕疵，而是将**机器可感知的法证级证据**（频域残差、局部异常、原型距离等）抽象为 Evidence Token 注入推理，实现“机器证据驱动的可信法证报告”。

| 论文 / 方向 | 任务范围 | 证据来源 | 解释形式 | 我们的可借鉴设计与切入点 |
| :--- | :--- | :--- | :--- | :--- |
| **FakeXplain** (ICLR 2026) | 通用 AIGC 图像检测 | 人工标注 BBox + Caption | 区域定位 + 文本解释 | 证明空间定位与文本解释结合可降低幻觉，启发我们将辅助法证分支输出转化为区域级 Evidence Token。 |
| **VERITAS** (ICLR 2026) | 人脸 Deepfake 检测 | MLLM 视觉观察 + 推理模式 | Fast / Planning / Reflection 结构化文本 | 借鉴其多阶段推理骨架（规划与反思对 OOD 泛化有帮助），改造为 Forensic-Aware Reasoning 链条。 |
| **AnomReason** (ICLR 2026) | 语义异常检测与推理 | 对象属性、关系、常识、物理异常 | 四元组 (Name/Phenomenon/Reasoning/Severity) | 提供语义异常证据结构，其 AnomAgent 可作为我们系统中的高层“Semantic Anomaly Expert”。 |
| **X-AIGD** (ICLR 2026) | 可解释检测 Benchmark | 像素级 Perceptual Artifact Mask | 低层、高层、认知三级伪迹 | 提醒我们“检测器强不等于解释可信”，需引入证据注意力对齐损失（Attention Alignment）约束解释的真实性。 |
| **我们的方案 (Proposed)** | **广义 AIGC 真实性检测** | **深度法证专家组 + 语义异常分支** | **机器证据驱动的结构化法证报告** | **构建主动取证机制，实现 MLLM 对机器法证特征的动态调用、多层证据融合与可信解释。** |

---

## 二、 为什么 MLLM 分析不到底层法证信息？

MLLM（如 GPT-4o, Gemini）由于其视觉预训练（CLIP-Vision）和网络架构的先天限制，在无辅助分支的情况下，根本无法感知以下关键特征：

1. **频域信息与高频伪迹 (Frequency-Domain Artifacts)**
   * **高频周期性网格 (Grid Artifacts)**：GAN 或 Diffusion 模型在反卷积/上采样（Upsampling）时，常会在 2D-FFT 频域留下规律的“指纹”斑点。
   * **高频残差谱与频谱截断**：图像经高通滤波后留下的噪声谱分布，以及生成图像在高频段不自然的抬升或突变。
2. **像素级微观统计分布 (Microscopic Pixel Statistics)**
   * **通道关联性异常 (Color Channel Correlations)**：真实相机在 RGB 三通道成像时存在特定的物理和透镜关联，生成图像的通道间噪声关联经常是失真的。
   * **局部噪声不一致 (Local Noise Inconsistency)**：局部拼接或编辑（Inpainting）区域的噪声方差（Noise Variance）与原图背景不一致，此特征在投影层（Projection Layer）中会完全丢失。
3. **压缩损耗与数字取证线索 (Compression Cues)**
   * **双重 JPEG 压缩痕迹 (Double JPEG)**：二次编辑并重新保存会使 DCT 系数直方图呈现出周期性的“挖空”效应。
   * **重采样与插值痕迹 (Resampling)**：图像拉伸、旋转后在像素差分空间留下的周期性协方差特征。
4. **图像分辨率的牺牲与降采样**
   * MLLM 通常将图像降采样至 $224 \times 224$ 或 $448 \times 448$ 以节省计算量，这一过程会直接抹杀绝大部分像素级的生成痕迹。

---

## 三、 主动探索型取证系统概念设计

基于上述痛点，项目早期提出了**“主动探索型（Active Exploration）双分支取证系统”**概念。其核心理念是：让 MLLM 扮演“法官”，通过语义分析初步锁定可疑区域后，主动、按需调用底层“法证专家组”，获取 Evidence Token，并在确信后适时终止、生成法证报告。当前代码只实现了其中的规则化原型，实际行为以架构文档为准。

```
                             [ MLLM 主分支 (法官) ]
                                  │       ▲
                        判定调取   │       │ 返回 Evidence Token (Δi)
                       (Action)   ▼       │
                             [ 法证专家组 (专家) ]
                             (频域 / 噪声 / 压缩等)
```

### 3.1 动态调取机制：不确定性引导的主动路由 (Active Routing)
为避免信息过载与注意力稀释，系统不采用“全量输入”方式，而是通过以下逻辑进行动态调取：

* **步骤 1：初始语义规划（Planning）**
  MLLM 对输入图像进行基础的视觉编码，并生成内部的思考链（Thinking Chain）：
  * *“图像逻辑通顺，但前景猫咪的边缘过渡似乎有些生硬，人眼无法确证。”*
* **步骤 2：触发动作 Token (Action Tool-Calling)**
  我们在 MLLM 的词表中定义特殊的动作 Token（如 `<call_freq>`, `<call_noise>`, `<call_jpeg>`）。当 MLLM 扫描到特定可疑区域（RoI）且自身视觉特征不确定时，它会输出一个动作指令和空间坐标：
  $$\text{Instruction} = \langle \text{call\_freq}, \text{bbox} = [x_1, y_1, x_2, y_2] \rangle$$
* **步骤 3：局部裁剪与特征抽象**
  系统捕获动作 Token 后，**裁剪**出指定 `bbox` 区域的图像送入对应的专家模型计算，并将输出抽象为结构化的 **Evidence Token ($\Delta_i$)** 喂回给 MLLM 的上下文。

#### 💡 调取的触发依据
* **语义与物理不确定性 (Semantic Uncertainty)**：检测到高层物理常识冲突（如阴影方向不一致，但无法通过像素确定是否合成），触发“局部噪声一致性专家”。
* **高频边缘与边界过渡 (Edge Anomalies)**：边缘交界处自注意力图（Attention Map）表现出异常高熵（模型感到困惑），自动触发“高频残差分析专家”。
* **全局重压缩疑虑 (Global Compression)**：当输入图像分辨率较低或有明显社交媒体传播痕迹时，率先触发“JPEG 压缩数学痕迹专家”。

---

## 四、 理论终止与生成机制（未完全实现）

### 4.1 什么时候停止调取？（Halting Criteria）
为了防止模型陷入无限循环调用，早期方案提出了基于信息熵与 KL 散度的动态终止目标。当前 `halting.py` 尚未实现该概率机制，只使用固定优先级和相邻 strength 差；替换步骤见 `plan.md` §4.10。

我们定义一个概率分布 $P(Y \mid I, \Delta_{1:t})$，表示在结合了前 $t$ 个 Evidence Token 后，模型对图像“真/伪/不确定”三分类的置信度。计算该分布的信息熵（Entropy）：
$$H(Y \mid I, \Delta_{1:t}) = - \sum_{y \in \{\text{Real}, \text{Fake}, \text{Uncertain}\}} P(y \mid I, \Delta_{1:t}) \log P(y \mid I, \Delta_{1:t})$$

1. **停止条件 1（置信收敛）**：当 $H(Y \mid I, \Delta_{1:t}) < \theta_h$（信息熵低于设定阈值，说明模型已经非常确信真伪）时，触发终止 Token 并输出 `<stop>`。
2. **停止条件 2（边际收益递减）**：如果新引入的证据 $\Delta_t$ 没有改变模型的判断（即前后两次概率分布的 KL 散度极小）：
   $$D_{KL} \left( P(Y \mid I, \Delta_{1:t}) \parallel P(Y \mid I, \Delta_{1:t-1}) \right) < \epsilon$$
   说明继续调取其他专家无助于消除疑惑，系统强制终止。
3. **停止条件 3（硬性上限预算）**：设置最大调用步数 $T_{\max} = 3$，达到上限后强制输出“Uncertain”并生成带有置信度校准的报告。

### 4.2 如何生成可信的“证据锚定”真伪判定？
为避免审稿人质疑“模型只是看了一眼检测器的分数，然后自己编了一段小作文（后置包装）”，我们设计了以下约束机制：

#### 1) 结构化证据表达 Schema
所有被调用的法证专家分支输出，必须被格式化为如下 Schema 才能被 MLLM 引用：
```json
{
  "evidence_name": "abnormal_high-frequency_residual",
  "region": "[x1, y1, x2, y2]",
  "phenomenon": "The boundary shows unusually concentrated residual responses.",
  "reasoning": "Associated with synthesis or post-generation blending artifacts.",
  "strength": 0.85,
  "source": "frequency-residual_branch",
  "support": "AI-generated",
  "uncertainty": "low"
}
```

#### 2) 推理模版约束与证据锚定（Evidence-Grounded Generation）
通过 SFT 或 GRPO 强化学习，约束 MLLM 的最终输出必须严格符合法证报告格式，且**生成文本中的每一个结论都必须链接到被调用的 Evidence Token 编号**：

> **[法证诊断结论]**: AI-Generated (置信度: 92%)
> **[逻辑推理链]**:
> 1. **观察（视觉层）**: 图像中杯子边缘存在异常虚化。
> 2. **法证证据引用**: 经调用 **`<call_freq>`** 专家，发现该区域在高频残差谱 $\Delta_1$ 中表现出明显的网格伪迹（特征强度: $0.85$），这与 GAN 模型的上采样指纹高度吻合。
> 3. **反思 (Reflection)**: 虽然物体物理光影逻辑合理，但底层微观特征已暴露出合成痕迹。

#### 3) 注意力-证据对齐损失 (Attention-Evidence Alignment Loss)
在训练 MLLM 生成解释文本时，设计**对齐约束**：
当模型在解码输出单词 `"网格伪迹"` 或 `"高频残差"` 时，其交叉注意力（Cross-Attention）权重必须高度集中在输入序列中的 $\Delta_1$（Evidence Token）以及对应的物理 BBox 上。如果模型一边写着“频域异常”，注意力却看着不相关的区域，则会受到**惩罚（Alignment Penalty）**。

---

## 五、 执行计划入口

本文原有的早期推进路线已经由代码实践、SFT 数据审计和新增论文对照进一步修正。为避免多份路线并行维护，后续不再在本文更新阶段顺序或待办；可信数据、坐标与证据协议、Expert v2、停止策略 v2、SFT v2、统一评测、GRPO 和局部篡改扩展均以 `plan.md` §4.7–§4.15 为准。
