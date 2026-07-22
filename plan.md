# 阶段一实现计划：主动探索型双分支图像取证系统原型

## Context

本项目从零构建一个"主动探索型 MLLM 图像取证系统"的原型。代码仓库当前只有设计文档（`.md`），无任何 Python 代码。阶段一目标：**在 CPU 模式下实现完整数据管道（真实专家算法 + Mock MLLM），用真实图像跑通端到端测试，并生成 SFT 训练数据**。

核心约束：无 GPU、纯 Python 状态机、操作日志增量追加。

---

## 一、目录结构

```
innovation_project/
├── main.py                         # 入口：加载配置 → 创建组件 → 运行状态机
├── config.py                       # 集中配置常量（路径、阈值、步数上限等）
├── requirements.txt                # 依赖清单
├── claude_operation_log.md         # 操作日志（已存在空文件，增量追加）
│
├── experts/                        # 法证专家模块
│   ├── __init__.py
│   ├── base.py                     # BaseExpert 抽象类 + ExpertResult dataclass
│   ├── frequency.py                # 2D-FFT 频域网格伪迹检测
│   ├── noise.py                    # SRM 高通滤波噪声残差分析
│   └── jpeg.py                     # JPEG 块效应 + DCT 直方图分析
│
├── mllm/                           # MLLM 客户端抽象层
│   ├── __init__.py
│   ├── base.py                     # BaseMLLMClient 抽象接口
│   └── mock_client.py             # 模板驱动的 Mock（按图像来源+轮次生成响应）
│
├── state_machine/                  # 状态机核心
│   ├── __init__.py
│   ├── controller.py              # 核心循环：解析→执行专家→Evidence Token→回传
│   ├── halting.py                 # 三重拦截守卫 + 信息熵/KL 散度计算
│   └── evidence_tokenizer.py      # 标量→语义描述映射 + Evidence Token Schema 构建
│
├── utils/                          # 工具层
│   ├── __init__.py
│   ├── parser.py                  # 正则解析器：抓取 <planning>/<call_*>/<reasoning>/<verdict>
│   ├── coordinate_transformer.py  # 相对坐标 [0,1000] → 绝对像素坐标
│   ├── image_utils.py            # 图像加载、bbox 裁剪、格式转换
│   └── logger.py                 # Trace Logger (ShareGPT 格式) + 操作日志追加
│
├── tests/                          # 测试
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures（合成图像、Mock 配置）
│   ├── test_parser.py
│   ├── test_coordinate_transformer.py
│   ├── test_image_utils.py
│   ├── test_frequency_expert.py
│   ├── test_noise_expert.py
│   ├── test_jpeg_expert.py
│   ├── test_evidence_tokenizer.py
│   ├── test_halting.py
│   ├── test_mock_mllm.py
│   ├── test_controller.py
│   └── test_pipeline.py          # 端到端管道测试（Real + GenImage 各 2 张图）
│
└── traces/                         # SFT 训练数据输出目录
    └── sft_sessions/              # 每次运行生成一个 ShareGPT JSON 文件
```

---

## 二、核心类/接口设计

### 2.1 专家模块 (`experts/`)

```python
# experts/base.py
@dataclass
class ExpertResult:
    """专家分析结果，直接映射到 Evidence Token Schema"""
    evidence_name: str          # e.g. "abnormal_high_frequency_residual"
    region: str                 # e.g. "patch_coordinates_[210,150,480,420]"
    phenomenon: str             # 物理现象描述
    reasoning: str              # 物理原理
    strength: float             # 归一化异常值 [0, 1]
    source: str                 # "frequency_expert" | "noise_expert" | "jpeg_expert"
    support: str                # "AI-generated" | "Real" | "Uncertain"
    interpretation_text: str    # 语义软描述

class BaseExpert(ABC):
    @abstractmethod
    def analyze(self, img_np: np.ndarray, bbox: list[int]) -> ExpertResult:
        """入参：BGR numpy 数组 + 绝对像素坐标 [ymin, xmin, ymax, xmax]"""
        ...
```

三个具体实现：

- **`FrequencyExpert`** — 裁剪 bbox → 灰度化 → Hanning 窗 → 2D-FFT → 功率谱 → 高频径向平均 → 峰值检测 → 归一化
- **`NoiseExpert`** — 裁剪 bbox → SRM 5×5 高通滤波核卷积 → 局部噪声方差 vs 全局背景方差 → 断层程度归一化
- **`JPEGExpert`** — 裁剪 bbox → 8×8 块效应强度 (Blockiness) → DCT 系数直方图 → 双重量化痕迹检测

### 2.2 MLLM 客户端 (`mllm/`)

```python
# mllm/base.py
class BaseMLLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, image_path: str,
                 history: list[dict]) -> str:
        """返回 MLLM 的原始文本输出（含 XML 标签）"""
        ...

# mllm/mock_client.py
class MockMLLMClient(BaseMLLMClient):
    """模板驱动 Mock：根据图像来源（real/fake）和对话轮次返回预设响应"""
    MODES = ["fast_verdict", "two_calls", "explore_all", "conflict"]
    def __init__(self, mode: str = "two_calls", seed: int = 42):
        ...
```

Mock 行为模式：

| 模式 | Turn 0 | Turn 1 | Turn 2+ | 用途 |
|------|--------|--------|---------|------|
| `fast_verdict` | planning + verdict（real）或 +1 call（fake） | reasoning + verdict | N/A | 测试快速结案路径 |
| `two_calls` | planning + 1 call | reasoning + 另1 call 或 verdict | verdict | 默认均衡测试 |
| `explore_all` | planning + 2 calls | reasoning + 2 calls | 持续至 max steps | 测试强制终止路径 |
| `conflict` | planning + call_freq | reasoning + call_noise | conflict → Uncertain | 测试证据冲突路径 |

### 2.3 状态机 (`state_machine/`)

```python
# state_machine/controller.py
class ForensicStateMachine:
    def __init__(self, mllm_client, experts, parser, tokenizer, halting, logger):
        ...
    def run(self, image_path: str, ground_truth: str = None) -> dict:
        """主循环，返回完整 Session 记录"""
        # 1. 加载图像 → conversation 初始化（含 System Prompt SOP 约束）
        # 2. while step < max_steps:
        #    a. MLLM.generate() → raw output
        #    b. Parser 提取 planning / call_* / reasoning / verdict
        #    c. if verdict → break
        #    d. if call_* → CoordinateTransformer → ImageUtils.crop → Expert.analyze
        #       → EvidenceTokenizer.build → 注入 conversation history
        #    e. HaltingChecker.check() → 可能强制终止
        # 3. Logger 保存完整 Session → SFT JSON
```

### 2.4 终止机制 (`state_machine/halting.py`)

```python
class HaltingChecker:
    def __init__(self, max_steps=5, entropy_threshold=0.3, kl_threshold=1e-3):
        ...
    def check(self, step, evidence_chain, last_output) -> (bool, str):
        """按优先级依次检查：verdict > max_steps > conflict > info_gain"""
    def _verdict_detected(self, output) -> str | None: ...
    def _max_steps_reached(self, step) -> bool: ...
    def _conflict_detected(self, evidence_chain) -> bool: ...
    def _info_gain_converged(self, evidence_chain) -> bool: ...
```

---

## 三、数据流（端到端）

```
1. 用户指定图像路径 → main.py
2. StateMachine.run(image_path)
3. Logger.init_sft_session() → 初始化 ShareGPT 数据结构
4. System Prompt 含 SOP 约束（planning/call/reasoning/verdict 格式要求）
5. 第 0 轮：MLLM.generate(system_prompt, image_path, history=[])
6. MockMLLM 返回：
   "<planning>
    Suspected Region: [200,150,400,350]
    Visual Anomalies: 边缘过于平滑
    Expert Target: freq — 检测上采样网格伪迹
    </planning>
    <call_freq>[200,150,400,350]</call_freq>"
7. Parser.extract_all_calls() → [("freq", [200,150,400,350])]
8. Parser.parse_verdict() → None → 继续
9. For each call:
   a. CoordinateTransformer.relative_to_absolute([200,150,400,350], W, H) → abs_bbox
   b. ImageUtils.crop_bbox(img, abs_bbox) → patch
   c. FrequencyExpert.analyze(patch) → ExpertResult(strength=0.82, support="AI-generated", ...)
   d. EvidenceTokenizer.tokenize(result, bbox, image_shape) → Evidence Token JSON
   e. conversation.append({"from": "user", "value": json.dumps(evidence_token)})
10. HaltingChecker.check() → (False, None) → 继续
11. 第 1 轮：MLLM.generate(prompt, image, updated_conversation)
12. MockMLLM 返回："<reasoning>...</reasoning><verdict>{"verdict":"Fake","confidence":0.92}</verdict>"
13. Parser.parse_verdict() → 提取 verdict → break
14. Logger.finalize_sft(verdict) → 保存 session JSON
15. Logger.log_operation() → 追加 claude_operation_log.md
```

---

## 四、关键设计决策

| 决策 | 选择 | 理由 |
|---|---|---|
| MLLM 接入 | 抽象接口 `BaseMLLMClient` | 当前 Mock 跑通管道，后续接真实 API 无需改状态机 |
| 状态机框架 | 纯 Python `while` 循环 | 解耦清晰，无框架依赖，符合任务书要求 |
| 专家算法 | 第一阶段即用真实算法（非 Mock） | SciPy/NumPy 纯 CPU，计算量小 |
| 坐标系统 | MLLM 输出 `[ymin, xmin, ymax, xmax]` × [0,1000] | 对齐 Qwen2.5-VL 规范 |
| BBox 顺序 | `[ymin, xmin, ymax, xmax]`（OpenCV 风格） | 与 NumPy 切片 `img[y0:y1, x0:x1]` 一致 |
| SFT 数据格式 | ShareGPT 多轮对话 JSON | 与任务文档 §7.2 Schema 对齐，可直接用于微调 |
| 图像内部格式 | BGR NumPy (OpenCV 格式) | 所有专家算法基于 OpenCV/SciPy |
| ADM RGBA 处理 | `cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)` | 集中处理于 `image_utils.py` |
| 小图像 (BigGAN 128×128) | bbox clip 保证最小 16×16 | `coordinate_transformer.py` 中处理 |

---

## 五、真实专家算法设计

### 5.1 Frequency Expert

```
输入: patch (BGR numpy array)
步骤:
  1. cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY) → gray
  2. gray * Hanning window → windowed
  3. np.fft.fft2(windowed) → fft_result
  4. np.fft.fftshift(fft_result) → centered
  5. power_spectrum = np.log(np.abs(centered) + 1)
  6. 高频径向平均（取外圈 50% 半径区域）
  7. 检测周期性尖峰：高频区域局部最大值超过中位数 + 3×std
  8. sigmoid 归一化 → strength ∈ [0, 1]
```

### 5.2 Noise Expert

```
输入: patch (BGR numpy array)
步骤:
  1. SRM 5×5 高通滤波核（空间富模型 kernel #1）
  2. cv2.filter2D(patch, -1, srm_kernel) → noise_residual (逐通道)
  3. 计算局部滑动窗口 (32×32) 的噪声方差
  4. 计算全局背景噪声方差
  5. 局部方差 / 全局方差 → 不一致性度量
  6. 断层程度归一化 → strength ∈ [0, 1]
```

### 5.3 JPEG Expert

```
输入: patch (BGR numpy array)
步骤:
  1. 计算水平/垂直块效应强度：
     - 检测 8×8 边界相邻像素差的周期性模式
     - B_h = mean(|I[i,8j] - I[i,8j-1]|) / mean(|I[i,j] - I[i,j-1]|)
  2. 对 8×8 块做 DCT (scipy.fft.dct)
  3. 检查 DCT 系数直方图是否有"挖空"效应（双重量化特征）
  4. 合并块效应 + 直方图异常 → 归一化 strength
```

---

## 六、Mock MLLM 模板设计

Mock 从预设模板库选择，不随机生成：

```
模板 TURN_0_REAL:
  <planning>
  Suspected Region: [250, 250, 750, 750]
  Visual Anomalies: 轻微压缩马赛克，疑似社交媒体传播痕迹
  Expert Target & Hypothesis: 拟调用 jpeg 专家，检测重压缩痕迹
  </planning>
  <call_jpeg>[250, 250, 750, 750]</call_jpeg>

模板 TURN_0_FAKE:
  <planning>
  Suspected Region: [200, 200, 800, 800]
  Visual Anomalies: 中心区域纹理过于平滑，缺乏真实相机噪点
  Expert Target & Hypothesis: 拟调用 freq 专家，生成图像在频域常有网格伪迹
  </planning>
  <call_freq>[200, 200, 800, 800]</call_freq>

模板 TURN_AFTER_EVIDENCE:
  <reasoning>
  【物理-语义一致性校验】底层 {expert_name} 反馈{evidence_strength}级异常：
  {phenomenon}——这与视觉观察到的{visual_anomaly}在因果链上吻合。
  </reasoning>
  [再调用另一个专家 或 输出 verdict]

模板 TURN_FINAL_FAKE:
  <reasoning>...</reasoning>
  <verdict>
  {"verdict": "Fake", "confidence": 0.92,
   "primary_evidence": "high_frequency_grid_artifact",
   "report": "经频域+噪声多轮法证分析，确认伪造。"}
  </verdict>

模板 CONFLICT:
  <reasoning>
  【证据冲突反思】freq 专家强判假(strength=0.9)，但 noise 专家强判真(strength=0.1)。
  底层物理痕迹出现不可调和的强冲突，不能做出确定结论。
  </reasoning>
  <verdict>
  {"verdict": "Uncertain", "confidence": 0.45,
   "report": "多项法证证据存在根本性冲突，疑罪从无。"}
  </verdict>
```

---

## 七、SFT 数据生成

每一次 `StateMachine.run()` 自动产出完整的 ShareGPT 格式多轮对话：

```json
{
  "id": "forensic_sft_20260717_001",
  "image_path": "dataset/GenImage_Test/Midjourney/0_midjourney_169.png",
  "ground_truth": "Fake",
  "source_model": "Midjourney",
  "final_verdict": {"verdict": "Fake", "confidence": 0.92, ...},
  "conversations": [
    {"from": "user", "value": "<image>\n请分析这张图像的真实性..."},
    {"from": "gpt", "value": "<planning>...</planning><call_freq>...</call_freq>"},
    {"from": "user", "value": "{\"evidence_name\":\"...\",\"strength\":0.82,...}"},
    {"from": "gpt", "value": "<reasoning>...</reasoning><verdict>...</verdict>"}
  ],
  "evidence_chain": [...],
  "metadata": {
    "image_size": [1024, 1024],
    "total_steps": 2,
    "halting_reason": "verdict_output",
    "mock_mode": "two_calls"
  }
}
```

输出路径：`traces/sft_sessions/session_{timestamp}_{image_name}.json`

---

## 八、实现顺序（依赖关系驱动）

```
阶段 1.1 ─ 基础设施（无依赖，可并行）
  ├── config.py
  ├── requirements.txt
  ├── utils/image_utils.py
  └── utils/logger.py

阶段 1.2 ─ 工具层（依赖 config）
  ├── utils/coordinate_transformer.py
  └── utils/parser.py

阶段 1.3 ─ 专家模块（依赖 base.py）
  ├── experts/base.py
  ├── experts/frequency.py    (依赖 scipy.fft)
  ├── experts/noise.py        (依赖 cv2)
  └── experts/jpeg.py         (依赖 numpy/scipy)

阶段 1.4 ─ MLLM 抽象层
  ├── mllm/base.py
  └── mllm/mock_client.py

阶段 1.5 ─ 状态机核心（依赖上述所有）
  ├── state_machine/evidence_tokenizer.py
  ├── state_machine/halting.py
  └── state_machine/controller.py

阶段 1.6 ─ 入口 + 测试
  ├── main.py
  └── tests/*
```

---

## 九、测试策略

### 单元测试
| 测试文件 | 测试内容 |
|---|---|
| `test_parser.py` | 正常/畸形 XML 标签匹配、多标签提取、verdict JSON 解析、空输入返回 None |
| `test_coordinate_transformer.py` | 往返转换精度、边界裁剪、0/1000 极值、非正方形图像 |
| `test_image_utils.py` | JPEG/PNG/RGBA 加载、bbox 裁剪、灰度转换 |
| `test_frequency_expert.py` | 合成网格图像→高 strength、均匀噪声→低 strength |
| `test_noise_expert.py` | 合成拼接图像→高 strength、自然图像→中低 strength |
| `test_jpeg_expert.py` | 单次 JPEG→低分、双重 JPEG→高分、PNG 无压缩→低分 |
| `test_evidence_tokenizer.py` | 强度映射边界（0.0/0.3/0.7/1.0）、Schema 完整性 |
| `test_halting.py` | 每种终止条件独立测试、无触发→返回 False |
| `test_mock_mllm.py` | 4 种模式 XML 结构正确、bbox 在 [0,1000] 范围、轮次追踪 |
| `test_controller.py` | 2 轮正常结案、max-step 终止、冲突终止、异常处理 |

### 端到端测试 (`test_pipeline.py`)
- Real × 2 张、Midjourney × 1 张、SD15 × 1 张
- 验证：管道不卡死、≥1 次专家调用、合法 verdict、SFT JSON 生成、操作日志追加

---

## 十、验证命令

```bash
pip install -r requirements.txt
python -m pytest tests/ -v                        # 全部测试
python -m pytest tests/test_pipeline.py -v        # 端到端
python main.py --image dataset/Real/xxx.jpg       # 单张手动运行
ls traces/sft_sessions/                           # SFT 数据检查
tail -30 claude_operation_log.md                  # 操作日志检查
```

---

## 十一、`requirements.txt`

```
numpy>=2.1.0
opencv-python>=5.0.0
scipy>=1.18.0
Pillow>=11.0.0
torch>=2.5.0
pytest>=8.0.0
```

## 十二、潜在风险与缓解

| 风险 | 缓解措施 |
|---|---|
| ADM 图像为 RGBA 256×256 | `image_utils.py` 集中处理 alpha → RGB 转换 |
| BigGAN 仅 128×128，bbox 过小 | `clip_bbox` 强制最小 16×16 |
| 不同专家归一化尺度不一致 | 各专家内部用 sigmoid 校准，而非 min-max |
| Mock MLLM 无法模拟真实视觉理解 | 测试关注 XML 结构+状态转移，不验证语义内容 |
| Parser 容错不够导致管道卡死 | 设计"格式纠错反馈环"：解析失败时注入错误提示重新生成 |

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

---

# 阶段三实现计划：SFT 监督微调、GRPO 对齐与专家重构

## Context

阶段二产出：816 条有效 ShareGPT SFT 数据 + 校准后的专家参数 + 真实 Qwen2.5-VL 推理管道。
阶段二验证：格式覆盖率 98%+，但端到端准确率仅 25%（Real=53%, Fake=20%）。

**基座 Qwen2.5-VL 根本问题**：
1. 不懂法证推理——收到 Evidence Token 后不知如何解读
2. 调用策略差——freq 被过度调用（64%）但其信号几乎为 0
3. 缺乏多轮意识——平均 1.9 steps 就结案，不会交叉验证
4. 不会反思——冲突场景下很少主动输出 Uncertain

**阶段三核心目标**：通过 SFT 微调注入法证推理能力 + GRPO 强化对齐固化行为模式 + 重构 freq 专家。

---

## 一、子阶段划分

### 3.1 SFT 监督微调

**目标**：用 816 条 ShareGPT 数据对 Qwen2.5-VL-7B 进行全参数或 LoRA 微调。

#### 3.1.1 数据预处理

- 从 `traces/sft_sessions/` 中筛选 verdict 非 null + evidence_chain 非空的样本
- 按 8:1:1 划分 train/val/test
- 确保 Real/Fake/Uncertain 三类 verdict 分布均衡
- 将 Qwen conversation 格式化为模型可接收的 messages 格式
- 输出标准化训练集到 `sft_data/train/`

#### 3.1.1b SFT 数据构造策略

**关键问题**：现有 816 条数据来自基座 Qwen 的实际输出——准确率仅 25%。直接用错误答案训练会强化错误。必须先构造**正确推理链**的示范数据。

**构造方法**：对每张图运行三专家获取真实 ExpertResult，然后基于 GT + 真实 Expert 输出 + 法证推理模板合成理想 Qwen 回复。四种互补类型：

| 类型 | 数量 | 构造方式 | 教什么 |
|------|------|----------|--------|
| **正确答案流** | ~400 | GT 与最强 Expert 信号一致的样本。合成从 planning→call→evidence→reasoning→verdict 的完整推理链，verdict 与 GT 一致 | 正常法证推理全流程 |
| **冲突反思流** | ~200 | 两个 Expert 信号方向相反（一强判假一强判真）的样本。合成双向反思 + 疑罪从无 + Uncertain 判定 + 置信度校准 | 证据冲突时的正确响应 |
| **格式示范流** | ~100 | 不依赖具体 Expert 输出。多种变体（不同 call 顺序、不同 bbox 位置），纯格式强化 | XML 标签条件反射 |
| **边界案例流** | ~100 | 选取 Expert strength 在 0.25-0.75 灰色地带的样本。合成谨慎推理 + 降置信度 | 灰色地带的谨慎判断 |

**类型一：正确答案流示例**（以 Midjourney 假图，GT=Fake 为例）：

实际 Expert 输出：
```
noise:   strength=0.76, AI-generated, "局部噪声方差 1.87，全局 5.43，塌陷率 65%"
freq_v2: strength=0.48, Uncertain, "高频段微弱网格 2.1σ，未达独立阈值"
jpeg:    strength=0.30, Real, "块效应 0.99，PNG 无压缩"  ← 忽略（信号弱）
```

合成的理想 Qwen 回复（3 轮）：
```
Turn 0 Qwen: <planning>
  人物面部纹理过度均匀，缺乏真实皮肤微纹理；背景与主体
  光照方向偏差约 15°。AI 生成模型倾向于产生这种过度平滑
  的纹理。优先调用 noise 专家验证局部噪声方差塌陷。
  → <call_noise>[380, 220, 720, 580]

Turn 1 Expert: noise → strength=0.76, AI-generated
  "局部噪声方差 1.87，远低于全局 5.43，塌陷率 65%"

Turn 2 Qwen: <reasoning>
  【物理-语义一致性校验】noise 专家检测到面部区域方差塌陷
  65%（strength=0.76）。与视觉观察到的"皮肤纹理过度均匀"
  在因果链上吻合——AI 渲染抹除了相机传感器微观噪声。
  
  【环境污染质询】PNG 格式无 JPEG 压缩干扰，65% 塌陷远超
  JPEG 混淆范围（通常 <15%），noise 信号可信度高。
  
  补充调取 freq 专家验证频域伪迹。
  → <call_freq>[380, 220, 720, 580]

Turn 3 Expert: freq_v2 → strength=0.48, Uncertain
  "微弱网格 2.1σ，方向与 AI 一致但未达独立阈值"

Turn 4 Qwen: <reasoning>
  【综合判定】
  noise: 方差塌陷 65%(0.76) → AI，可信度高
  freq:  微网格 2.1σ(0.48) → 方向一致，辅助确认
  两条证据均指向 AI，无证据支持真实相机来源。
  → <verdict>Fake(0.86)
```

**类型二：冲突反思流示例**（GT=Fake，但 Expert 信号矛盾）：

实际 Expert 输出：
```
freq:  strength=0.91, AI-generated, "频域尖峰 4.2σ，典型的 GAN 上采样网格"
noise: strength=0.08, Real, "噪声方差 4.12 vs 全局 4.08，偏差 < 1%"
```

合成的理想 Qwen 回复（最终 verdict 为 Uncertain）：
```
<reasoning>
【证据冲突分析】
freq 专家强烈指出 AI 生成（4.2σ 网格，strength=0.91），
但 noise 专家强烈指出真实相机（偏差 < 1%，strength=0.08）。
两条证据在物理层面出现不可调和的根本性冲突。

【冲突溯源】
可能解释：(1) 真实照片经 AI 锐化/超分辨率处理——保留原始
噪声结构但引入 AI 上采样伪迹；(2) AI 生成图像叠加了后处理
噪声——freq 检测到生成痕迹但 noise 被掩盖。

【疑罪从无】当前证据不足以做出确定性判定。
→ <verdict>Uncertain(0.45)
```

**类型三：格式示范流示例**（纯格式训练，不关心对错）：

该类型提供多种格式变体——首轮 call noise、首轮 call jpeg、单 call 即结案、双 call 后结案等——确保模型在任意情况下都能输出正确的 XML 标签结构。推理内容模板化，重点在于标签闭合、顺序正确。

**类型四：边界案例流示例**（GT=Real，noise strength=0.32）：

```
Expert 输出: noise strength=0.32, Uncertain, "方差偏差 18%，处于灰色地带"
GT: Real

合成 Qwen 回复:
<reasoning>
18% 的方差偏差处于真实相机自然波动（<15%）和明显异常
（>30%）之间的灰色地带。自然场景中的光照变化和纹理复杂度
差异也可能导致类似偏差。在无其他专家强信号的情况下，
倾向判定为真实，但置信度相应降低。
→ <verdict>Real(0.72)
```

#### 3.1.1c 当前数据状态与待改进项

**已产出**：365 条合成数据（correct=165, conflict=200），borderline=0, format=0。

**三个根本问题与改进方案**：

| # | 问题 | 改进方案 | 优先级 |
|---|------|----------|--------|
| 1 | 模板 reasoning 公式化——"填空式写作"，非真正法证推理 | 从阶段二 A 线数据中筛选 verdict 与 GT 一致的样本（~25%×610≈140 条），将其作为"真实正确推理"数据；手写 10 条高质量 ideal reasoning 作为风格种子 | **高** |
| 2 | Expert 信号方向错误——GenImage PNG 假图的 noise/jpeg 偏 Real | 在合成模板中显式加入"格式差异分析"段落（"该图为 PNG 格式，无 JPEG 压缩历史，noise/jpeg 信号受格式影响"），让模型学会区分格式伪影和 AI 伪影 | **高** |
| 3 | borderline + format 类型缺失 | 从现有 390 张基准集中选取 strength 在 [0.25,0.6] 的样本生成 borderline（~100 条）；从任意样本生成 format 类型（~100 条），纯 XML 标签格式强化 | **中** |

**改进后目标数据分布**：

| 类型 | 目标数量 | 来源 |
|------|----------|------|
| 正确答案流 | ~300 | 165 合成 + 140 筛选自 A 线 |
| 冲突反思流 | ~200 | 已就绪 |
| 边界案例流 | ~100 | 待生成 |
| 格式示范流 | ~100 | 待生成 |
| **总计** | **~700** | Ready for SFT |

#### 3.1.1d Expert reasoning 修复与数据重生成评估 (2026-07-21)

**修复**：四个专家（`frequency`/`noise`/`jpeg`/`frequency_v2`）的 `reasoning` 字段从硬编码模板改为三段式条件化输出。此前无论 strength=0.01 还是 0.9，reasoning 永远输出"这是 AI 生成特征"——导致 Evidence Token 内部自相矛盾，误导 Qwen。

**波及范围评估**：

| 数据文件 | 来源 | 受旧 reasoning 影响？ | 需要重生成？ | 需要 GPU？ |
|----------|------|----------------------|-------------|-----------|
| `sft_correct.json` (196) | A 线筛选，真实 Qwen 输出 | Evidence Token 中包含旧 reasoning，但 Qwen 的最终 verdict=GT | ⚠️ 不理想但可用 | 是 |
| `sft_conflict.json` (200) | `build_sft_data.py` 合成 | 合成模板中嵌入了旧 Expert reasoning | ⚠️ 同上 | 否 (CPU) |
| `sft_borderline.json` (100) | `finalize_sft_data.py` 合成 | 同上 | **是** (CPU 几分钟) | 否 |
| `sft_format.json` (100) | A 线抽取 | 格式训练不看内容 | ❌ 不需要 | — |

**决策**：
- borderline **立即重生成**（CPU，几分钟）——数据量小，影响直接
- correct/conflict **暂不重生成**——verdict 与 GT 一致，训练目标正确；Evidence Token 中的旧 reasoning 恰好模拟真实场景中 Expert 信号不完美的情形
- GPU 开启后如有剩余时间，可选重跑 A 线部分样本作为对比

#### 3.1.2 训练配置

| 参数 | 建议值 | 说明 |
|------|--------|------|
| 框架 | LLaMA-Factory / transformers Trainer | 前者更便捷，后者更灵活 |
| 微调方式 | LoRA (rank=64, alpha=128) | 节省显存，24 GB 可承载 |
| 学习率 | 2e-5 | 标准 SFT 学习率 |
| Batch size | 2 (gradient accumulation ×4) | 有效 batch=8 |
| Epochs | 3 | 避免过拟合 |
| Max length | 2048 | 覆盖多轮对话 |
| GPU | RTX 4090 × 1 (24 GB) | LoRA 模式足够 |

#### 3.1.3 训练目标

- **格式硬收敛**：`<planning>/<call_*>/<reasoning>/<verdict>` 标签错漏率 < 0.1%
- **多轮状态感应**：模型能分辨"首轮看图→中轮收证据→末轮结案"的阶段职责
- **法证推理注入**：学会将 Evidence Token 的定性描述与视觉观察交叉关联

#### 3.1.4 实现内容

- `scripts/prepare_sft_data.py`：数据清洗 + train/val/test 划分 + Qwen 格式转换
- `scripts/train_sft.py` 或 `sft_config.yaml`：LLaMA-Factory 训练配置
- 训练完成后保存 LoRA adapter 到 `checkpoints/sft_lora/`

### 3.2 专家算法重构

**目标**：解决阶段二发现的两个核心问题。

#### 3.2.1 Frequency Expert 重设计

**当前状态**：raw_metric 恒 ~0，分离度 0.05，完全无效。

**根因分析**：
- 当前算法在 bbox 区域（通常 200×200~400×400）上做 2D-FFT
- GenImage 图像多为 PNG，无压缩伪迹，且生成质量高
- 高频周期性峰值检测在中小 patch 上分辨率不足

**改进方案**：
- 对**全图**而非 bbox crop 做 FFT（阶段二校准已证明全图分析可行）
- 增加**多尺度 FFT**（在不同分辨率下检测）
- 引入**预训练 CNN 分类器**作为替代方案（在 GenImage 上训练一个轻量 ResNet-18 频域特征提取器）
- 保持 `BaseExpert` 接口不变，替换内部实现

#### 3.2.2 专家调用策略优化

**当前状态**：freq 被 Qwen 调用 64%，但其信号为 0——浪费推理预算。

**改进方案**：
- 在 System Prompt 中增加**调用指南**：明确告诉模型 freq 适用于哪些场景、noise/jpeg 适用于哪些场景
- 可选：在状态机层面增加**智能路由**——用简单的图像特征（分辨率、格式、压缩率）预判应优先调哪个专家
- 训练数据中**增加不同专家调用顺序的多样性**

#### 3.2.3 实现内容

- `experts/frequency_v2.py`：重写的频域专家（全图 FFT + 多尺度）
- `scripts/calibrate_experts_v2.py`：更新校准脚本，验证新 freq 的分离度
- `config.py`：更新 freq 参数和 System Prompt 调用指南

### 3.3 GRPO 强化学习对齐

**目标**：使用规则奖励对 SFT 后的模型进行组内相对策略优化，固化行为模式。

#### 3.3.1 Reward 设计（来自任务书 §8）

| Reward | 权重 | 触发条件 |
|--------|------|----------|
| **Format** | +0.2 / -0.5 | 严格遵循 SOP 标签格式 |
| **Anti-Numerical Laziness** | +0.5 / -0.6 | reasoning 中引用了定性描述词（如 "grid residual"）而非仅提分数 |
| **Attention-Evidence Consistency** | +0.4 | verdict 声称异常的区域与 call 的 bbox 有空间一致性（IoU > 0.5） |
| **Outcome Accuracy** | +1.0 / -1.0 | 分类正确；对 Uncertain +0.5 额外奖励 |

#### 3.3.2 实现内容

- `scripts/grpo_reward.py`：实现 4 个 Reward 函数的计算逻辑
- 奖励计算依赖状态机 Trace Log（已在阶段一实现）
- 与 SFT 后的模型组成 GRPO 训练循环
- 输出：GRPO 对齐后的模型权重

**注意**：GRPO 需要 RL 训练框架（如 TRL 的 GRPOTrainer），且需要 GPU 长时间运行。此子阶段可作为 SFT 之后的进阶优化，不一定是阶段三的硬性交付。

### 3.4 全数据集评估与论文素材

**目标**：在完整 GenImage 测试集上评估最终系统性能，产出论文级指标。

#### 3.4.1 评估指标

- **分类准确率**：Real vs Fake 二分类 + Real/Fake/Uncertain 三分类
- **按生成模型细分**：ADM/BigGAN/Glide/Midjourney/SD14/SD15/VQDM/Wukong 各子类准确率
- **法证报告质量**（人工评估子集）：
  - 证据-结论一致性
  - 物理-语义交叉质证完整性
  - 环境污染分析的覆盖度
- **消融实验**：
  - 仅 MLLM（无专家）vs 单专家 vs 双专家 vs 三专家
  - Mock vs SFT 后 vs GRPO 后
  - 校准前 vs 校准后的专家参数

#### 3.4.2 实现内容

- `scripts/evaluate_full.py`：全数据集批量评估脚本
- `scripts/ablation.py`：消融实验脚本
- 输出评估报告（JSON + 可视化图表）

---

## 二、目录结构变更

```
innovation_project/
├── experts/
│   └── frequency_v2.py              # [NEW] 重写的频域专家
│
├── checkpoints/                      # [NEW] 模型权重
│   ├── sft_lora/                    # SFT LoRA adapter
│   └── grpo/                        # GRPO 对齐权重
│
├── sft_data/
│   └── train/                       # [NEW] 标准化训练集
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── scripts/
│   ├── prepare_sft_data.py          # [NEW] SFT 数据预处理
│   ├── train_sft.py                 # [NEW] SFT 训练脚本
│   ├── grpo_reward.py               # [NEW] GRPO Reward 函数
│   ├── evaluate_full.py             # [NEW] 全数据集评估
│   └── ablation.py                  # [NEW] 消融实验
│
├── evaluation/                       # [NEW] 评估报告
│   ├── full_eval_report.json
│   └── ablation_report.json
│
└── config.py                        # [MODIFIED] 更新 freq 参数和 System Prompt
```

---

## 三、实现顺序

```
阶段 3.1a ─ 数据预处理（CPU，可立即开始）
  ├── scripts/prepare_sft_data.py
  ├── 数据清洗 + 8:1:1 划分
  └── Qwen messages 格式转换

阶段 3.2 ─ 专家重构（CPU，可并行于 3.1a）
  ├── experts/frequency_v2.py
  ├── scripts/calibrate_experts_v2.py
  └── System Prompt 调用指南更新

阶段 3.1b ─ SFT 训练（GPU，依赖 3.1a）
  ├── scripts/train_sft.py / LLaMA-Factory 配置
  ├── LoRA 微调（~4-6 小时 RTX 4090）
  └── 保存 adapter → checkpoints/sft_lora/

阶段 3.3 ─ GRPO 对齐（GPU，依赖 3.1b，可选）
  ├── scripts/grpo_reward.py
  ├── GRPO 训练循环
  └── 保存 GRPO 权重 → checkpoints/grpo/

阶段 3.4 ─ 全数据集评估（GPU，依赖 3.1b + 3.2）
  ├── scripts/evaluate_full.py
  ├── scripts/ablation.py
  └── 论文指标 + 图表
```

---

## 四、关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 微调方式 | LoRA (rank=64) | 24 GB VRAM 可承载，训练快，可回滚 |
| 训练框架 | LLaMA-Factory | 支持 Qwen2.5-VL，开箱即用 |
| freq 重构 | 全图 FFT + 多尺度 → 最终替换为轻量 CNN | 逐级升级，保持接口不变 |
| GRPO | SFT 后可选 | SFT 是硬交付，GRPO 是学术加分项 |
| 评估基线 | 阶段二未训练的 Qwen2.5-VL 作为 baseline | 量化 SFT 的提升幅度 |

---

## 五、GPU 时间预估

| 子阶段 | 预估时间 | 是否需要 GPU |
|--------|---------|-------------|
| 3.1a 数据预处理 | ~10 min | CPU |
| 3.2 专家重构 | ~30 min | CPU |
| 3.1b SFT 训练 | ~4-6 hours | GPU |
| 3.3 GRPO 对齐 | ~8-12 hours | GPU |
| 3.4 全数据集评估 | ~2-3 hours | GPU |

---

## 六、阶段三完成标准

- [ ] SFT 训练数据预处理完成（train/val/test 划分，格式标准化）
- [ ] LoRA 微调完成，loss 收敛，格式错漏率 < 0.5%
- [ ] Frequency Expert v2: 分离度 > 0.3（当前 0.05）
- [ ] 端到端准确率 > 50%（当前 25%），至少翻倍
- [ ] 消融实验完成：SFT 后 vs SFT 前、单专家 vs 多专家
- [ ] 法证报告质量人工评估：证据-结论一致性 > 80%
- [ ] 操作日志完整

---

## 七、阶段三本质变化

> **阶段二证明了"管道能跑通真模型"** —— 真实 Qwen2.5-VL 的推理行为被忠实地记录下来。
>
> **阶段三实现"模型能用法证"** —— 通过 SFT 微调将 816 条真实推理样本中的模式注入模型，
> 使其学会：(1) 如何解读 Evidence Token 的物理含义，(2) 何时应该交叉验证而非轻信单个专家，
> (3) 如何在证据冲突时退回到 Uncertain 并给出置信度校准。最终产出一个具备法证推理能力的
> 专用 Qwen2.5-VL 变体。
