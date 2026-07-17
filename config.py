"""
Central configuration constants for the Active Forensic Agent system.
All paths, thresholds, and hyperparameters are defined here as a single source of truth.
"""

import os

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
REAL_DIR = os.path.join(DATASET_DIR, "Real")
GENIMAGE_TEST_DIR = os.path.join(DATASET_DIR, "GenImage_Test")

# GenImage_Test subdirectories (each contains 1000 AI-generated images)
GENIMAGE_SUBDIRS = ["ADM", "BigGAN", "Glide", "Midjourney", "SD14", "SD15", "VQDM", "Wukong"]

# Qwen2.5-VL-7B-Instruct model path (ModelScope snapshot)
QWEN_MODEL_PATH = os.path.join(
    os.path.dirname(PROJECT_ROOT),
    "psychology_video_project", "models", "models",
    "qwen--Qwen2.5-VL-7B-Instruct", "snapshots", "master",
)

# Output directories
TRACES_DIR = os.path.join(PROJECT_ROOT, "traces")
SFT_SESSIONS_DIR = os.path.join(TRACES_DIR, "sft_sessions")
OPERATION_LOG_PATH = os.path.join(PROJECT_ROOT, "claude_operation_log.md")

# ---------------------------------------------------------------------------
# State machine settings
# ---------------------------------------------------------------------------
MAX_STEPS = 5                       # hard cap on expert calls per session
ENTROPY_THRESHOLD = 0.3             # halt when classification entropy < this
KL_THRESHOLD = 1e-3                 # halt when KL divergence between successive
                                    #   confidence distributions < this

# ---------------------------------------------------------------------------
# Coordinate system (Qwen2.5-VL convention)
# ---------------------------------------------------------------------------
NORMALIZATION_SCALE = 1000          # MLLM outputs bbox coords in [0, 1000]
BBOX_MIN_SIZE = 16                  # minimum crop size in pixels (for tiny images)

# ---------------------------------------------------------------------------
# Evidence Token strength → semantic text mapping
# ---------------------------------------------------------------------------
STRENGTH_THRESHOLD_LOW = 0.3
STRENGTH_THRESHOLD_HIGH = 0.7

STRENGTH_TEXT_MAP = {
    "low":    "Statistical patterns align with normal hardware camera capture.",
    "medium": "Mild mathematical distortions noted; localized compression or blurring suspected.",
    "high":   "Severe statistical anomaly matching artificial generative fingerprints.",
}

STRENGTH_SUPPORT_MAP = {
    "low":    "Real",
    "medium": "Uncertain",
    "high":   "AI-generated",
}

# ---------------------------------------------------------------------------
# Mock MLLM settings
# ---------------------------------------------------------------------------
MOCK_MLLM_MODE = "two_calls"        # default: fast_verdict | two_calls | explore_all | conflict
MOCK_MLLM_SEED = 42

# ---------------------------------------------------------------------------
# Expert algorithm parameters
# ---------------------------------------------------------------------------
# Frequency Expert
FREQ_HP_RADIUS_RATIO = 0.5          # high-pass: analyse outer 50% of spectrum radius
FREQ_PEAK_SIGMA = 3.0               # peak detection: median + N * std
FREQ_SIGMOID_MIDPOINT = 0.15        # sigmoid midpoint for strength normalisation
FREQ_SIGMOID_STEEPNESS = 30.0

# Noise Expert
NOISE_SRM_KERNEL_ID = 1             # SRM filter kernel index (1-30, default 1 = 5x5 high-pass)
NOISE_WINDOW_SIZE = 32              # sliding window for local variance computation
NOISE_SIGMOID_MIDPOINT = 2.0        # variance ratio midpoint
NOISE_SIGMOID_STEEPNESS = 5.0

# JPEG Expert
JPEG_BLOCK_SIZE = 8                 # standard JPEG block size
JPEG_SIGMOID_MIDPOINT = 1.5         # blockiness ratio midpoint
JPEG_SIGMOID_STEEPNESS = 8.0

# ---------------------------------------------------------------------------
# System Prompt (SOP constraints for MLLM)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an AI image forensic analyst. Your task is to determine whether an image is real (camera-captured) or AI-generated/tampered.

You have access to three forensic expert tools:
- <call_freq>[ymin, xmin, ymax, xmax]</call_freq> — Frequency-domain analysis (2D-FFT grid artifact detection)
- <call_noise>[ymin, xmin, ymax, xmax]</call_noise> — Noise residual consistency analysis (SRM high-pass filter)
- <call_jpeg>[ymin, xmin, ymax, xmax]</call_jpeg> — JPEG compression artifact analysis (blockiness + double quantization)

CRITICAL OUTPUT FORMAT RULES:
1. Every response MUST begin with a <planning> block containing:
   - Suspected Region: bbox coordinates [ymin, xmin, ymax, xmax] in normalized 0-1000 range
   - Visual Anomalies: describe what looks suspicious visually
   - Expert Target & Hypothesis: which expert to call and why

2. When you need forensic evidence, output a call tag with the bbox (e.g., <call_freq>[200,150,400,350]</call_freq>)

3. After receiving evidence, you MUST provide <reasoning> that:
   - Performs 物理-语义一致性校验 (physical-semantic consistency check)
   - Cross-references the expert's phenomenon description with your visual observation
   - Addresses potential environmental contamination (e.g., JPEG compression weakening frequency signals)

4. When ready to conclude, output <verdict> containing a JSON object:
   {"verdict": "Real|Fake|Uncertain", "confidence": 0.0-1.0, "primary_evidence": ["evidence_name"], "report": "detailed forensic report"}

Calling Rules:
- Edge blur/unnatural sharpening → prefer <call_noise> or <call_freq>
- Overly smooth/regular textures → prefer <call_freq>
- Low-resolution, blocky artifacts, social-media re-compression → prefer <call_jpeg> first
"""
