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
