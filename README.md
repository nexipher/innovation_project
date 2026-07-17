# Forensic-Agent

**Active Exploration Dual-Branch Image Forensic System**  
MLLM-driven, evidence-grounded AI-generated image detection with explainable forensic reports.

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.5-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)
[![Tests](https://img.shields.io/badge/tests-95%20passed-brightgreen)](./tests/)

---

## Overview

Modern Multi-modal Large Language Models (MLLMs) excel at high-level semantic understanding but are fundamentally blind to low-level forensic signals — frequency-domain artifacts, micro-scale noise patterns, and JPEG compression traces — because their visual encoders (e.g., CLIP) discard this information during downsampling.

**Forensic-Agent** bridges this gap by combining an MLLM "judge" with a team of traditional forensic "experts" in an active exploration loop:

1. The MLLM inspects the image and identifies suspicious regions
2. It dynamically invokes forensic experts (`freq` / `noise` / `jpeg`) on specific bounding boxes
3. Expert outputs are abstracted into structured **Evidence Tokens** with physical-semantic descriptions
4. The MLLM cross-examines all evidence and produces a reasoned, evidence-grounded verdict

> **Key Insight:** _FakeXplain teaches MLLMs to reason over human-annotated visible artifacts, while our method teaches MLLMs to reason over machine-extracted forensic evidence._

---

## Architecture

```
                    [Input Image]
                         │
        ┌────────────────┴────────────────┐
        ▼                                 ▼
  [MLLM "Judge"]  ◄───────────  [State Machine Controller]
  (semantic analysis               │
   + decision making)              │  Action Tokens
        │                          │  <call_freq/noise/jpeg>[bbox]
        │ SOP output               ▼
        │                  [Forensic Experts]
        │                  ├── FrequencyExpert  (2D-FFT grid artifacts)
        │                  ├── NoiseExpert      (SRM noise residual)
        │                  └── JPEGExpert       (blockiness + double quant.)
        │                          │
        │                          │ Evidence Tokens (structured JSON)
        ▼                          ▼
  [Evidence-Grounded Verdict + SFT Training Data]
```

### Project Structure

```
innovation_project/
├── main.py                         # CLI entry point
├── config.py                       # Centralized constants, thresholds, system prompt
├── requirements.txt                # Dependencies
│
├── experts/                        # Forensic expert modules (CPU-friendly)
│   ├── base.py                     # BaseExpert ABC + ExpertResult dataclass
│   ├── frequency.py                # 2D-FFT power spectrum peak detection
│   ├── noise.py                    # SRM high-pass filter + variance inconsistency
│   └── jpeg.py                     # Blockiness metric + DCT double-quantization
│
├── mllm/                           # MLLM client abstraction layer
│   ├── base.py                     # BaseMLLMClient abstract interface
│   └── mock_client.py              # Template-driven mock (4 behaviour modes)
│
├── state_machine/                  # Core orchestration
│   ├── controller.py               # Main while-loop: parse → execute → inject → halt
│   ├── halting.py                  # 4-tier halting guards (verdict/max/info_gain/conflict)
│   └── evidence_tokenizer.py       # ExpertResult → Evidence Token Schema JSON
│
├── utils/                          # Tool layer
│   ├── image_utils.py              # Load / crop / grayscale / format conversion
│   ├── coordinate_transformer.py   # [0,1000] rel ↔ abs pixel (Qwen2.5-VL convention)
│   ├── parser.py                   # Regex-based XML tag extraction + validation
│   └── logger.py                   # SessionLogger (ShareGPT SFT) + operation log
│
├── tests/                          # Test suite (95 tests)
│   ├── conftest.py                 # Shared fixtures (synthetic images, mock clients)
│   ├── test_parser.py              # 18 tests
│   ├── test_pipeline.py            # 12 E2E tests (Real/MJ/SD15/ADM)
│   ├── test_halting.py             # 14 tests
│   ├── test_coordinate_transformer.py  # 10 tests
│   ├── test_image_utils.py         # 8 tests
│   ├── test_mock_mllm.py           # 6 tests
│   ├── test_controller.py          # 6 tests
│   ├── test_evidence_tokenizer.py  # 6 tests
│   ├── test_frequency_expert.py    # 5 tests
│   ├── test_noise_expert.py        # 5 tests
│   └── test_jpeg_expert.py         # 4 tests
│
├── traces/sft_sessions/            # Generated SFT training data (ShareGPT JSON)
└── claude_operation_log.md         # Development audit log
```

---

## Quick Start

### Prerequisites

- Python 3.12+
- CPU-only mode is fully supported (no GPU required for experts)

### Install

```bash
pip install -r requirements.txt
```

### Run a Single Image

```bash
python main.py --image dataset/Real/002baac0-bacd-496c-981c-a4a9d66b8472.jpg
python main.py --image dataset/GenImage_Test/Midjourney/0_midjourney_169.png
```

### Batch Mode

```bash
# Run all Real images (limit 5)
python main.py --batch Real --max 5

# Run all Midjourney images
python main.py --batch Midjourney

# Run all categories (Real + 8 GenImage subdirs)
python main.py --batch all --max 10
```

### Behaviour Modes

```bash
python main.py --image path/to/img.png --mode fast_verdict   # 2 turns, quick conclusion
python main.py --image path/to/img.png --mode two_calls      # 2 turns, balanced (default)
python main.py --image path/to/img.png --mode explore_all    # exhaust all 3 experts
python main.py --image path/to/img.png --mode conflict       # creates contradictory evidence
```

### Run Tests

```bash
pytest tests/ -v              # All 95 tests
pytest tests/test_pipeline.py # End-to-end only
```

---

## Pipeline Output

Running `main.py` produces:

### 1. Console Verdict

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

### 2. SFT Training Data (ShareGPT Format)

Each run generates a multi-turn conversation JSON under `traces/sft_sessions/`:

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

Ready for Qwen2.5-VL supervised fine-tuning.

---

## Forensic Experts

| Expert | Algorithm | What It Detects | CPU |
|--------|-----------|-----------------|-----|
| **Frequency** (`freq`) | Hanning window → 2D-FFT → power spectrum → radial high-pass → periodic peak detection | GAN/Diffusion upsampling grid artifacts | ✓ |
| **Noise** (`noise`) | SRM 5×5 high-pass kernel → local variance vs. global → inconsistency scoring | Splicing / inpainting / AI local editing | ✓ |
| **JPEG** (`jpeg`) | 8×8 block-boundary gradient ratio + DCT coefficient histogram hollowing detection | Double JPEG compression / re-save traces | ✓ |

Each expert outputs an **Evidence Token** with physical phenomenon descriptions and normalized anomaly scores (0–1), preventing the MLLM from "number-only" lazy reasoning.

---

## Halting Criteria

| Priority | Criterion | Trigger |
|----------|-----------|---------|
| 1 | **Verdict Output** | MLLM produces `<verdict>` tag |
| 2 | **Max Steps** | ≥ 5 expert-call iterations |
| 3 | **Evidence Conflict** | One expert strongly says Fake (strength > 0.7), another strongly says Real (strength < 0.3) |
| 4 | **Information Gain** | KL divergence between successive confidence distributions < threshold |

---

## SFT Data Pipeline

The system is designed from the ground up for supervised fine-tuning:

- Every pipeline run automatically generates a **ShareGPT-format** multi-turn conversation JSON
- Full evidence chain with physical phenomenon descriptions is preserved
- Metadata includes: ground truth, source model, halting reason, mock mode
- Ready to feed into Qwen2.5-VL SFT training (Phase 2)

---

## Stage 1 Achievements

| Capability | Status |
|---|---|
| End-to-end pipeline (image → verdict) | ✅ |
| 3 real forensic expert algorithms (CPU) | ✅ |
| 4 Mock MLLM behaviour modes | ✅ |
| 4-tier halting mechanism | ✅ |
| SFT data generation (ShareGPT JSON) | ✅ |
| Coordinate system (Qwen2.5-VL [0,1000] spec) | ✅ |
| Full test suite (95 tests, 100% pass) | ✅ |
| Operation audit log | ✅ |

### Current Limitations

- **MLLM is Mock**: Template-driven responses; does not perform real visual understanding
- **Frequency Expert sensitivity**: Low on small cropped regions; sigmoid calibration needed
- **No real SFT yet**: Training data is generated, but Qwen2.5-VL fine-tuning is Phase 2

---

## Roadmap

| Phase | Scope |
|-------|-------|
| **1.1–1.6** (done) | Infrastructure, experts, MLLM abstraction, state machine, tests |
| **2** | Real Qwen2.5-VL API integration (GPU required); frequency expert calibration |
| **3** | SFT training on generated data; GRPO reinforcement alignment |
| **4** | Multi-region attention-evidence alignment; full dataset evaluation |

---

## Environment

Developed and tested on **AutoDL Cloud Platform**:
- **GPU**: NVIDIA RTX 4090 (24 GB) — available on-demand
- **CPU mode**: Default for development (no GPU cost)
- **Base image**: Ubuntu 22.04 / Python 3.12 / PyTorch 2.5.1 / CUDA 12.4

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{forensic-agent,
  author = {HJ},
  title  = {Forensic-Agent: Active Exploration Dual-Branch Image Forensic System},
  year   = {2026},
  note   = {MLLM-driven, evidence-grounded AI-generated image detection},
}
```

## License

MIT
