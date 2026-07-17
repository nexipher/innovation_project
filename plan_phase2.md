# 阶段二实现计划：真实 MLLM 接入、SFT 数据规模化与专家校准

## Context

阶段一已完成：纯 CPU 管道跑通（Mock MLLM + 真实专家算法）、95 个测试全部通过、SFT 数据生成机制就绪。

阶段二核心目标：**将 Mock MLLM 替换为真实 Qwen2.5-VL，规模化生成高质量 SFT 训练数据，校准专家算法参数，为阶段三的监督微调做好准备。**

关键变化：需要 GPU 环境（按 agent.md §2.2 规范，须先获得用户明确授权）。

---

## 一、阶段二目标

| 目标 | 阶段一状态 | 阶段二目标 |
|------|-----------|-----------|
| MLLM | Mock 模板驱动 | 真实 Qwen2.5-VL API 调用 |
| 运行环境 | 纯 CPU | GPU（RTX 4090），按需开启 |
| SFT 数据 | 机制就绪，零星生成 | 规模化生成（目标 1000+ 条） |
| 专家算法 | 基础实现，参数未校准 | Frequency 敏感度校准 + 三专家阈值对齐 |
| 行为模式 | 4 种 Mock 模板 | 真实模型的多轮 Tool-calling 行为 |
| 格式容错 | Parser 基础校验 | 真实模型的格式纠错反馈环实战验证 |

---

## 二、子阶段划分

### 2.1 真实 Qwen2.5-VL 接入

**目标**：实现 `QwenVLClient`，替换 `MockMLLMClient`，保持 `BaseMLLMClient` 接口不变。

**实现内容**：

- `mllm/qwen_client.py`：继承 `BaseMLLMClient`
  - 支持两种推理后端：
    - **A) 本地 vLLM 部署**（需要 GPU，端口 6006）
    - **B) API 调用**（DashScope / OpenAI 兼容 API）
  - 图像预处理：按 Qwen2.5-VL 规范 resize + normalize
  - 多轮对话上下文管理（System Prompt + 图像 + Evidence Token 历史）
  - 超时重试 + 格式校验自动重生成
- 更新 `mllm/__init__.py` 注册新客户端
- 更新 `main.py` 支持 `--mllm qwen` 参数切换

**格式纠错反馈环**（对应任务书 §6 防"数值懒惰"）：

- 解析 Qwen 输出 → Parser 校验
- 若缺失必要标签 → 注入纠错提示 → 重新生成（最多 2 次）
- 若 verdict JSON 格式错误 → 注入修复提示 → 重新生成
- 记录重试次数到 session metadata

**GPU 审批**：在加载模型权重前，先向用户报告预计显存占用（Qwen2.5-VL-7B ~16 GB FP16）、单张推理时间（~2-5s）、任务必要性。

### 2.2 SFT 数据规模化生成

**目标**：批量运行管道，生成高质量、多样化的 ShareGPT 训练数据。

**数据构造策略**（对应任务书 §7.2）：

- **A 线 — 正常破案流**（占总数据 70%）：
  - Real 图像 200 张 × GenImage 每类 50 张
  - 1-3 轮工具调用后成功结案
  - 覆盖 4 种行为模式的真实模型输出
- **B 线 — 拦截与冲突流**（占总数据 30%）：
  - 刻意触发 max_steps / info_gain / conflict
  - 包含系统强制终止提示词和模型的反思响应

**数据质量控制**：

- 自动过滤：verdict 缺失 / 格式不闭合 / 标签嵌套错误的样本直接丢弃
- 多样性检查：确保 3 个专家被调用的频次大致均衡
- 标注 ground truth 与 pipeline verdict 的对照表

**实现内容**：

- `scripts/generate_sft_data.py`：批量数据生成脚本
  - 支持 `--stream A` / `--stream B` / `--stream both`
  - 支持 `--num-samples 1000` 控制总量
  - 实时进度条 + 错误统计
- 输出目录结构：
  ```
  sft_data/
  ├── stream_a_normal/     # 正常破案流
  ├── stream_b_conflict/   # 拦截与冲突流
  └── metadata.json        # 数据集统计信息
  ```

### 2.3 专家算法校准

**目标**：调整三个专家的 sigmoid 参数和检测阈值，使其在真实图像上的区分度最大化。

**问题诊断**（阶段一发现）：

- Frequency Expert：在 crop 区域上 strength 普遍 < 0.05，sigmoid_midpoint 过于保守
- Noise Expert：raw_metric 范围 0.3-1.5，sigmoid_midpoint=2.0 过高，导致所有结果被压缩到 < 0.2
- JPEG Expert：行为正确——Real 图像（JPEG）的 trace 明显，GenImage 图像（PNG）无压缩痕迹

**校准方案**：

1. **基准测试集**：Real 100 张 + GenImage 每类 30 张 = 340 张
2. **全图分析**：不再局限 bbox crop，对全图运行各专家（提高 FFT 分辨率 + 噪声估计精度）
3. **ROC 分析**：在 Real vs Fake 二分类上扫描 sigmoid 参数
4. **确定最优参数**：
   - 各 sigmoid_midpoint 使 Youden 指数最大化
   - `STRENGTH_THRESHOLD_LOW/HIGH` 使三分类准确率最大化
5. 更新 `config.py` 中的默认参数

**实现内容**：

- `scripts/calibrate_experts.py`：
  - 批量运行三专家 → 收集 raw_metric 分布
  - 网格搜索最优 sigmoid/threshold 参数
  - 输出校准报告（JSON）

### 2.4 格式纠错反馈环实战验证

**目标**：验证真实 Qwen2.5-VL 的输出格式稳定性。

**验证指标**：

- 首次输出格式正确率（目标 > 85%）
- 需要纠错重试的比例
- 纠错后成功率（目标 > 98%）
- verdict JSON 可解析率

---

## 三、目录结构变更

```
innovation_project/
├── mllm/
│   └── qwen_client.py          # [NEW] 真实 Qwen2.5-VL 客户端
│
├── scripts/                     # [NEW] 批量脚本目录
│   ├── generate_sft_data.py    # SFT 数据规模化生成
│   └── calibrate_experts.py    # 专家参数校准
│
├── sft_data/                    # [NEW] 规模化 SFT 数据输出
│   ├── stream_a_normal/
│   ├── stream_b_conflict/
│   └── metadata.json
│
├── calibration/                 # [NEW] 校准结果
│   └── calibration_report.json
│
└── config.py                    # [MODIFIED] 更新校准后的参数
```

---

## 四、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Qwen 推理后端 | 优先本地 vLLM，备选 DashScope API | 批量生成数据时本地推理无 API 费用 |
| 格式纠错 | 最多重试 2 次，失败则丢弃 | 避免无限循环；坏样本不应进入 SFT 数据集 |
| SFT 数据规模 | 目标 1000-2000 条 | 参考 LLaMA-Factory 等框架的最小可用 SFT 集 |
| 专家校准 | 基于全图而非 bbox | bbox 区域过小导致 FFT 分辨率和方差估计不准确 |
| GPU 使用 | 仅在 SFT 数据生成和推理时开启 | 符合 agent.md 成本控制原则 |

---

## 五、实现顺序

```
阶段 2.1 ─ 真实 MLLM 接入（需 GPU 审批）
  ├── mllm/qwen_client.py
  ├── 格式纠错反馈环
  └── main.py --mllm qwen 切换

阶段 2.2 ─ 专家校准（可并行于 2.1，CPU 可跑校准脚本的数据收集部分）
  ├── scripts/calibrate_experts.py
  ├── 基准测试集构建
  └── config.py 参数更新

阶段 2.3 ─ SFT 数据规模化生成（依赖 2.1 + 2.2）
  ├── scripts/generate_sft_data.py
  ├── A 线 + B 线数据生成
  └── 数据质量报告

阶段 2.4 ─ 验证与评估
  ├── 格式正确率统计
  ├── SFT 数据质量审查
  └── 端到端性能基准（准确率、平均步数、终止原因分布）
```

---

## 六、测试策略

- `tests/test_qwen_client.py`：Mock Qwen API 响应的单元测试（格式纠错环 + 重试逻辑验证）
- `tests/test_calibrate.py`：校准脚本在少量图像上的功能测试
- `tests/test_sft_quality.py`：SFT 数据 Schema 校验、标签完整性检查

---

## 七、验证方式

```bash
# 单张真实 MLLM 推理（需 GPU）
python main.py --image dataset/Real/xxx.jpg --mllm qwen

# 专家校准（CPU 可跑数据收集部分）
python scripts/calibrate_experts.py --num-real 100 --num-fake 30

# SFT 数据规模化生成（需 GPU）
python scripts/generate_sft_data.py --stream both --num-samples 1500

# 数据质量报告
python scripts/generate_sft_data.py --report-only
```

---

## 八、阶段二完成标准

- [ ] `QwenVLClient` 可正常调用，输出通过 Parser 校验
- [ ] 格式纠错反馈环实战有效（纠错后成功率 > 98%）
- [ ] 三专家 sigmoid 参数经 ROC 校准，Real vs Fake 区分度显著提升
- [ ] 生成 ≥ 1000 条高质量 SFT 数据（A 线 + B 线）
- [ ] SFT 数据通过 Schema 校验和多样性检查
- [ ] 全部测试通过
- [ ] 操作日志完整

---

## 九、阶段一 vs 阶段二：输出对比

以同一张 Midjourney 图像 `0_midjourney_169.png` 为例。

### 阶段一输出（现在 — Mock MLLM）

```
============================================================
Image:   dataset/GenImage_Test/Midjourney/0_midjourney_169.png
GT:      Fake
Mode:    two_calls
============================================================

  Verdict:    Fake
  Confidence: 0.9206
  Steps:      1
  Halting:    verdict_output
  Evidence:   1 expert(s) called
    - frequency_expert: strength=0.0110 → Real
  SFT data:   traces/sft_sessions/session_xxx.json
```

**阶段一的根本问题**：

- Mock 没有真正"看"图。Planning 写 "unnaturally smooth textures" 是模板固定文本——无论哪张 Fake 图都输出同一句话
- Evidence 显示 frequency=0.01（完全没检测到异常），但 Mock 仍然判 Fake(0.92)
- **证据和结论脱节**：MLLM 的 reasoning 不依赖 Expert 的实际输出，而是按模板拼接

### 阶段二完成后（理论输出 — 真实 Qwen2.5-VL）

```
============================================================
Image:   dataset/GenImage_Test/Midjourney/0_midjourney_169.png
GT:      Fake
MLLM:    Qwen2.5-VL-7B (vLLM)
Mode:    two_calls
============================================================

  Verdict:    Fake
  Confidence: 0.87
  Steps:      2
  Halting:    verdict_output
  Evidence:   2 expert(s) called
    - noise_expert:      strength=0.7630 → AI-generated
    - frequency_expert:  strength=0.4200 → Uncertain
  SFT data:   traces/sft_sessions/session_xxx.json
  MLLM retries: 0
```

### 逐维度对比

| 维度 | 阶段一（Mock） | 阶段二（真实 Qwen2.5-VL） |
|------|---------------|--------------------------|
| **Planning** | 模板固定文本，所有 Fake 图一样 | 模型真实观察图像，描述*这张图具体*的视觉异常 |
| **专家选择** | 模板预设（Fake→freq, Real→jpeg） | 模型根据视觉观察*自主决定*：看到过平滑纹理→优先调 noise，看到边缘锯齿→调 freq |
| **BBox 定位** | 固定坐标（中心 70%） | 模型锁定*实际可疑区域*的像素坐标 |
| **证据解读** | 模板拼接 strength 数值 | 真实交叉质证：将 noise 的"方差塌陷"与视觉上的"过度平滑"在因果链上关联 |
| **环境污染分析** | 模板一句话带过 | 具体分析：如"该图为 PNG 格式，无 JPEG 压缩污染，noise 信号可信度高" |
| **Verdict 置信度** | 伪随机 0.82-0.95 | 由证据链真实支撑，冲突时主动降置信度 |
| **专家校准** | freq 几乎永远 0.01（无效） | 校准后 freq 敏感度提升，三个专家各司其职 |
| **SFT 数据质量** | 模板拼接的"假数据" | 真实模型推理行为的忠实记录，可直接用于微调 |

### 阶段二理论输出：接近真实学术场景的完整交互

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MLLM 第 0 轮 ─ Planning:

  Suspected Region: [180, 220, 820, 780]
  Visual Anomalies: 前景主体与背景交界处存在不自然的羽化过渡；
                    皮肤纹理在高倍放大后呈现水彩画式的均匀平滑，
                    缺乏真实相机传感器的微观噪点颗粒感。
  Expert Target: 优先调用 noise 专家 — 验证局部噪声方差是否存在塌陷。

→ 状态机调用 NoiseExpert(bbox=[184, 225, 839, 798])
→ 返回 Evidence Token:
   strength=0.76, support=AI-generated
   phenomenon: "该区域噪声方差为 1.87，远低于全图背景方差 5.43，
               方差塌陷率约 65%，与 AI 局部重绘/羽化操作一致"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MLLM 第 1 轮 ─ Reasoning + 二次调用:

  【物理-语义一致性校验】
  noise 专家在该区域检测到显著的噪声方差塌陷（strength=0.76）。
  这与我在视觉层观察到的"边缘羽化"和"皮肤纹理过平滑"在因果链上高度吻合——
  AI 重绘/羽化操作会抹除原始传感器噪声并引入均匀合成纹理。

  【环境污染质询】
  该图为 PNG 格式（无 JPEG 压缩），不存在压缩污染导致噪声信号衰减的干扰因素，
  noise 专家的 0.76 置信度可信度较高。

  为进一步确证，补充调取 freq 专家检测高频段是否存在上采样网格伪迹。

→ 调用 FrequencyExpert(bbox=[184, 225, 839, 798])
→ 返回: strength=0.42, support=Uncertain

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

MLLM 第 2 轮 ─ Verdict:

  【综合判定】
  - noise 专家：局部方差塌陷 65%（强判假，可信度高）
  - freq 专家：高频段存在微弱网格状能量聚集，但未达显著阈值（判 Uncertain）
  - 综合来看，噪声层面的物理指纹断层是确定性证据，频域信号提供了辅助线索。

  <verdict>
  {
    "verdict": "Fake",
    "confidence": 0.87,
    "primary_evidence": ["noise_residual_inconsistency"],
    "report": "图像前景区域经噪声残差分析确认存在显著的局部方差塌陷（65%），
              与AI后处理（局部重绘/边缘羽化）的物理特征一致。频域分析发现辅助性
              线索但未达独立判定阈值。综合判定为 AI 生成/篡改图像。"
  }
  </verdict>
```

### 阶段二本质变化

> **阶段一证明了"管道能跑"** —— Mock MLLM + 真实专家 + 状态机的工程可行性。
>
> **阶段二实现了"管道能用"** —— 真实 MLLM 看图、自主决策、交叉质证、
> 生成真正证据锚定的法证报告。生成的 SFT 数据不再是模板拼接的假数据，
> 而是真实模型推理行为的忠实记录，可直接用于阶段三的监督微调训练。
