# Forensic-Agent

**主动探索型双分支图像取证系统**  
MLLM 驱动、法证证据锚定的 AI 生成图像检测，输出可解释的法证报告。

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-95%20passed-brightgreen)](./tests/)

---

## 项目背景

多模态大语言模型（MLLM，如 GPT-4o、Qwen2.5-VL）长于高层语义理解，却对频域伪迹、微观噪声分布、JPEG 压缩痕迹等底层物理法证特征处于"睁眼瞎"状态——其视觉编码器（CLIP）在下采样过程中丢弃了这些关键信息。

**本系统**将 MLLM 作为"法官"，与传统法证"专家组"结合，构建主动探索式闭环：

1. MLLM 扫描图像，锁定可疑区域
2. 按需动态调用法证专家（`freq` / `noise` / `jpeg`），传入具体 bbox
3. 专家输出被抽象为结构化 **Evidence Token**（含物理现象描述 + 置信度）
4. MLLM 交叉质证全部证据，生成证据锚定的可解释真伪判定报告

> **核心差异**：FakeXplain 教模型推理人类标注的可见伪迹，而本方法教模型推理机器提取的法证级证据。

---

## 系统架构

```
                    [输入图像]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
  [MLLM "法官"]   ◄───────────  [状态机控制器]
  (语义分析+决策)                    │
        │                            │ Action Token
        │ SOP 结构化输出              │ <call_freq/noise/jpeg>[bbox]
        ▼                            ▼
                          [法证专家组]
                          ├── 频域专家 (2D-FFT 网格伪迹)
                          ├── 噪声专家 (SRM 残差一致性)
                          └── JPEG专家 (块效应+双重量化)
                                    │
                                    │ Evidence Token (JSON)
                                    ▼
                  [证据锚定的真伪判定 + SFT 训练数据]
```

### 项目结构

```
innovation_project/
├── main.py                         # CLI 入口
├── config.py                       # 集中配置常量、阈值、System Prompt
├── requirements.txt                # 依赖清单
│
├── experts/                        # 法证专家模块（纯 CPU）
│   ├── base.py                     # BaseExpert 抽象类 + ExpertResult 数据结构
│   ├── frequency.py                # 2D-FFT 功率谱峰值检测
│   ├── noise.py                    # SRM 高通滤波 + 局部方差不一致性
│   └── jpeg.py                     # 块效应度量 + DCT 双重量化检测
│
├── mllm/                           # MLLM 客户端抽象层
│   ├── base.py                     # BaseMLLMClient 抽象接口
│   └── mock_client.py              # 模板驱动 Mock（4 种行为模式）
│
├── state_machine/                  # 状态机核心
│   ├── controller.py               # 主循环：解析→执行专家→Evidence Token 注入→终止判断
│   ├── halting.py                  # 四重终止守卫（verdict/max_steps/conflict/info_gain）
│   └── evidence_tokenizer.py       # ExpertResult → Evidence Token Schema JSON
│
├── utils/                          # 工具层
│   ├── image_utils.py              # 图像加载 / 裁剪 / 灰度化 / 格式转换
│   ├── coordinate_transformer.py   # 相对坐标 [0,1000] ↔ 绝对像素坐标
│   ├── parser.py                   # 正则解析器（XML 标签提取 + 结构化校验）
│   └── logger.py                   # SessionLogger (ShareGPT SFT) + 操作审计日志
│
├── tests/                          # 测试套件（95 个用例）
│   ├── conftest.py                 # 共享 fixtures（合成图像、Mock 配置）
│   ├── test_parser.py              # 18 个
│   ├── test_pipeline.py            # 12 个（端到端：Real/MJ/SD15/ADM）
│   ├── test_halting.py             # 14 个
│   ├── test_coordinate_transformer.py  # 10 个
│   ├── test_image_utils.py         # 8 个
│   ├── test_mock_mllm.py           # 6 个
│   ├── test_controller.py          # 6 个
│   ├── test_evidence_tokenizer.py  # 6 个
│   ├── test_frequency_expert.py    # 5 个
│   ├── test_noise_expert.py        # 5 个
│   └── test_jpeg_expert.py         # 4 个
│
├── traces/sft_sessions/            # SFT 训练数据输出（ShareGPT 格式 JSON）
└── claude_operation_log.md         # 开发操作审计日志
```

---

## 快速开始

### 环境要求

- Python 3.12+
- 纯 CPU 模式完全可用（专家算法无需 GPU）

### 安装

```bash
pip install -r requirements.txt
```

### 单张图像分析

```bash
python main.py --image dataset/Real/002baac0-bacd-496c-981c-a4a9d66b8472.jpg
python main.py --image dataset/GenImage_Test/Midjourney/0_midjourney_169.png
```

### 批量分析

```bash
# 分析 Real 子集（限制 5 张）
python main.py --batch Real --max 5

# 分析全部 Midjourney 图像
python main.py --batch Midjourney

# 分析全部类别
python main.py --batch all --max 10
```

### 行为模式

```bash
python main.py --image path/to/img.png --mode fast_verdict   # 2 轮快速结案
python main.py --image path/to/img.png --mode two_calls      # 2 轮均衡分析（默认）
python main.py --image path/to/img.png --mode explore_all    # 遍历 3 个专家
python main.py --image path/to/img.png --mode conflict       # 制造证据冲突
```

### 运行测试

```bash
pytest tests/ -v              # 全部 95 个测试
pytest tests/test_pipeline.py # 仅端到端测试
```

---

## 管道输出

### 1. 控制台判定

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

### 2. SFT 训练数据（ShareGPT 格式）

每次运行在 `traces/sft_sessions/` 下生成一份多轮对话 JSON：

```json
{
  "id": "session_20260717_124042_0_midjourney_169",
  "ground_truth": "Fake",
  "source_model": "Midjourney",
  "final_verdict": {"verdict": "Fake", "confidence": 0.9206},
  "conversations": [
    {"from": "user", "value": "<image>\n请分析这张图像的真实性..."},
    {"from": "gpt",  "value": "<planning>...</planning><call_freq>...</call_freq>"},
    {"from": "user", "value": "{\"evidence_name\":\"...\",\"strength\":0.011,...}"},
    {"from": "gpt",  "value": "<reasoning>...</reasoning><verdict>...</verdict>"}
  ],
  "evidence_chain": [...]
}
```

可直接用于 Qwen2.5-VL 监督微调（SFT）。

---

## 法证专家

| 专家 | 算法 | 检测目标 | CPU |
|------|------|----------|-----|
| **频域专家** (`freq`) | Hanning 窗 → 2D-FFT → 功率谱 → 高频径向峰值检测 | GAN/Diffusion 上采样网格伪迹 | ✓ |
| **噪声专家** (`noise`) | SRM 5×5 高通滤波核 → 局部方差 vs 全局方差 → 不一致性评分 | 拼接 / AI 局部重绘 / 边缘羽化 | ✓ |
| **JPEG 专家** (`jpeg`) | 8×8 块边界梯度比 + DCT 系数直方图"挖空"检测 | 双重 JPEG 压缩 / 二次保存痕迹 | ✓ |

每个专家输出 **Evidence Token**，包含物理现象的自然语言描述和归一化异常值（0–1），防止 MLLM"只看数值不看物理含义"的惰性推理。

---

## 终止机制

| 优先级 | 条件 | 触发逻辑 |
|--------|------|----------|
| 1 | **模型主动结案** | MLLM 输出 `<verdict>` 标签 |
| 2 | **最大步数封顶** | 专家调用 ≥ 5 轮 |
| 3 | **证据强冲突** | 一专家强判假（strength > 0.7），另一专家强判真（strength < 0.3） |
| 4 | **信息增益收敛** | 连续两轮 Evidence Token 置信度变化低于阈值 |

---

## SFT 数据管道

系统从底层设计即面向监督微调：

- 每次管道运行自动生成 **ShareGPT 格式**多轮对话 JSON
- 完整保留证据链（物理现象、原理解释、置信度）
- 元数据包含：ground truth、生成模型来源、终止原因、Mock 模式
- 可直接作为 Qwen2.5-VL SFT 阶段的训练样本（阶段二）

---

## 阶段一成果

| 能力 | 状态 |
|------|------|
| 端到端管道（图像 → 判定） | ✅ |
| 3 个真实法证专家算法（纯 CPU） | ✅ |
| 4 种 Mock MLLM 行为模式 | ✅ |
| 四重终止机制 | ✅ |
| SFT 训练数据生成（ShareGPT JSON） | ✅ |
| 坐标系统（Qwen2.5-VL [0,1000] 规范） | ✅ |
| 完整测试套件（95 用例，100% 通过） | ✅ |
| 操作审计日志 | ✅ |

### 当前局限

- **MLLM 为 Mock**：决策逻辑基于模板，非真实视觉理解
- **频域专家敏感度偏低**：在小尺寸裁剪区域上不明显（sigmoid 参数需校准）
- **尚未进行真实 SFT**：训练数据已生成，Qwen2.5-VL 微调属于阶段二工作

---

## 路线图

| 阶段 | 范围 |
|------|------|
| **1.1–1.6**（已完成） | 基础设施、专家算法、MLLM 抽象、状态机、测试 |
| **二** | 接入真实 Qwen2.5-VL API（需 GPU）；频域专家校准 |
| **三** | SFT 监督微调 + GRPO 强化学习对齐 |
| **四** | 多区域注意力-证据对齐；全数据集评估 |

---

## 运行环境

基于 **AutoDL 算力云平台** 开发与测试：

- **GPU**：NVIDIA RTX 4090 (24 GB) — 按需开启
- **CPU 模式**：日常开发默认模式，零 GPU 费用
- **基础镜像**：Ubuntu 22.04 / Python 3.12 / PyTorch 2.5.1 / CUDA 12.4

---

## 引用

```bibtex
@misc{forensic-agent,
  author = {HJ},
  title  = {Forensic-Agent: 主动探索型双分支图像取证系统},
  year   = {2026},
  note   = {MLLM-driven, evidence-grounded AI-generated image detection},
}
```

## 许可证

MIT
