#!/usr/bin/env python3
"""
Active Forensic Agent — CLI entry point.

Usage:
    python main.py --image path/to/image.jpg
    python main.py --image path/to/image.jpg --mode conflict
    python main.py --batch Real            # run all Real images
    python main.py --batch Midjourney      # run all Midjourney images
    python main.py --batch all --max 10    # run 10 images from each category
"""

import argparse
import os
import sys
from datetime import datetime

from config import (
    REAL_DIR,
    GENIMAGE_TEST_DIR,
    GENIMAGE_SUBDIRS,
    MOCK_MLLM_MODE,
)
from mllm.message_builder import build_forensic_prompt
from mllm.mock_client import MockMLLMClient

# 两个真实客户端都在函数里导入：mock 路径（CPU、测试、dry-run）不该为
# torch+transformers 付导入代价（曾把导入基线从 116 MB 抬到 586 MB）。
# 专家的导入放在函数里：npr/univfd 模块会拉 torch，而 mock 路径（CPU、测试、
# dry-run）不该为 GPU 栈付导入代价——这条不变量由 tests/test_g8_entry.py 守着。
from state_machine.controller import ForensicStateMachine
from utils.logger import SessionLogger, log_operation


def build_probe_expert(mllm_type: str):
    """
    PROBE when it is usable, None when it is not (G7).

    Its code and weights are not shipped with this project, so a box without
    them must still start; the tool is simply absent from the registry and
    from the prompt.  CPU sessions get the mock, which is what makes the
    whole PROBE path testable without a GPU.
    """
    # CPU 会话用 mock 专家；`qwen`（Qwen3）与 `qwen25`（历史复算）都是真会话。
    if mllm_type == "mock":
        from experts.probe_dino import MockProbeExpert
        return MockProbeExpert()
    from experts.probe_dino import ProbeDinoExpert
    expert = ProbeDinoExpert()
    reason = expert.unavailable_reason()
    if reason:
        # 不是 Mock 替换（不会造假分数），但**主检测器缺席**必须说清楚：这次会话
        # 的工具箱只剩 npr/univfd，任何"g8 工具箱"的结论都不适用于它。
        print(f"Warning: PROBE 不可用（{reason}）。本次会话不提供主检测器 PROBE，"
              "工具箱只剩 npr/univfd——该会话的结果不得按 g8 工具箱口径解读。")
        return None
    return expert


def require_cuda_for_real_sessions(mllm_type: str) -> None:
    """
    真实入口的第一道门：没有可用的 GPU 就**拒绝**，而不是换成 Mock。

    曾经的写法是 `print("Warning: CUDA not available. Falling back to
    MockMLLMClient.")` 然后继续跑（用户 2026-10-15 点名禁止）：产出的 trace、
    证据链与落盘记录与真实会话无异，验收会把它当成真结果——"看起来跑通"里最贵
    的一种。这道门放在**加载任何权重之前**：拒绝要快，不留半个已初始化的会话。
    """
    import torch                          # 只有真会话分支才需要它

    if torch.cuda.is_available():
        return
    raise RuntimeError(
        f"--mllm {mllm_type} 需要 CUDA，但这台机器上没有可用的 GPU。\n"
        "真实会话不会退回 Mock（那会让验收看起来跑通）：\n"
        "  * 想在 CPU 上跑协议/管道：显式用 `--mllm mock`\n"
        "  * 想跑真实模型：换到有 GPU 的机器，或申请 GPU 后再跑")


def build_pipeline(mllm_type: str = "mock", mode: str = MOCK_MLLM_MODE,
                   greedy: bool = False, inference_mode: str = "non-thinking",
                   local_files_only: bool = False,
                   tools=None) -> ForensicStateMachine:
    """
    Construct the forensic pipeline with an explicit tool policy.

    Default remains the stage-4.4 toolbox (probe / npr / univfd; freq/jpeg/noise
    are replay-only).  Passing ``tools=("probe", "trufor")`` builds the minimal
    joint-task loop: one whole-image expert and one localisation expert, with no
    silent substitution.  A real session never drops a requested tool.

    解码条件（`greedy`/`inference_mode`/`local_files_only`）走这里传入，而不是让
    调用方自己造一个等价客户端：验收、冒烟与批次生成必须是**同一条构造路径**——
    否则"我看的那批"与"跑出来的那批"会在没人注意的地方分叉（默认是采样解码，
    温度 0.1；批次与冒烟都用贪心，便于复现）。
    """
    desired = (list(tools) if tools is not None
               else list(ForensicStateMachine.CURRENT_TOOL_TAGS))
    known = set(ForensicStateMachine.EXPERT_KEY_MAP)
    unknown = [tag for tag in desired if tag not in known]
    if unknown:
        raise RuntimeError(f"未知工具：{unknown}")
    if not desired:
        raise RuntimeError("工具箱为空：至少指定一个工具")
    if mllm_type in ("qwen", "qwen25"):
        require_cuda_for_real_sessions(mllm_type)   # 先拒，再加载任何权重

    experts = {}
    if "probe" in desired:
        probe = build_probe_expert(mllm_type)
        if probe is not None:
            experts["probe_dino_expert"] = probe

    for name in ("npr", "univfd"):
        if name not in desired:
            continue
        if mllm_type == "mock":
            if name == "npr":
                from experts.npr import MockNPRExpert
                experts["npr_expert"] = MockNPRExpert()
            else:
                from experts.univfd import MockUnivFDExpert
                experts["univfd_expert"] = MockUnivFDExpert()
        else:
            if name == "npr":
                from experts.npr import NPRExpert
                expert = NPRExpert()
            else:
                from experts.univfd import UnivFDExpert
                expert = UnivFDExpert()
            reason = expert.unavailable_reason()
            if reason:
                raise RuntimeError(f"{expert.source_name} 不可用：{reason}\n"
                                   "真实会话不缩减工具箱——请修好它，或用显式的其他策略跑。")
            experts[expert.source_name] = expert
            print(f"专家设备：{expert.source_name}={expert.device}")

    if "trufor" in desired:
        if mllm_type == "mock":
            from experts.trufor import MockTruForExpert
            experts["trufor_expert"] = MockTruForExpert(device="cpu")
        else:
            from experts.trufor import TruForExpert
            expert = TruForExpert(device="cuda")
            reason = expert.unavailable_reason()
            if reason:
                raise RuntimeError(f"{expert.source_name} 不可用：{reason}\n"
                                   "双专家会话必须有局部定位器，不能静默缩减工具箱。")
            experts[expert.source_name] = expert
            print(f"专家设备：{expert.source_name}={expert.device}")

    available = {ForensicStateMachine.EXPERT_SOURCE_MAP[name]
                 for name in experts if name in ForensicStateMachine.EXPERT_SOURCE_MAP}
    missing = [tag for tag in desired if tag not in available]
    # The historical default (tools=None) kept running when PROBE weights were
    # absent and warned; only an EXPLICIT policy is refused, because a caller
    # that named its tools must not receive a silently smaller toolbox.
    if missing and tools is not None:
        raise RuntimeError(
            f"请求的工具没有注册成功：{missing}。真实/显式工具箱不做静默降级；"
            "请修复权重或改用显式的其他策略。")
    tags = [tag for tag in desired if tag in available]
    system_prompt = build_forensic_prompt(tags)

    if mllm_type in ("qwen", "qwen25"):
        if mllm_type == "qwen":
            from mllm.qwen3_client import Qwen3VLClient
            mllm = Qwen3VLClient(system_prompt=system_prompt,
                                 inference_mode=inference_mode,
                                 do_sample=not greedy,
                                 temperature=0.0 if greedy else 0.1,
                                 local_files_only=local_files_only)
            revision = getattr(mllm, "revision", None)
            print(f"Using {mllm.name}" + (f" @ {revision[:8]}" if revision else "") + " (GPU)")
            if callable(getattr(mllm, "identity", None)):
                print(f"  model identity: {mllm.identity()}")
        else:
            from mllm.qwen_client import QwenVLClient
            from config import QWEN_MODEL_PATH
            if not os.path.exists(QWEN_MODEL_PATH):
                raise RuntimeError(
                    f"--mllm qwen25 需要 Qwen2.5-VL 权重，但 {QWEN_MODEL_PATH} 不存在。\n"
                    "该模型仅供**历史复算**；g8 主线请用 `--mllm qwen`（Qwen3-VL）。")
            print("Using Qwen2.5-VL-7B-Instruct (GPU, 仅历史复算)")
            mllm = QwenVLClient(system_prompt=system_prompt)
    else:
        mllm = MockMLLMClient(mode=mode)

    print(f"工具策略：{tags}")
    return ForensicStateMachine(mllm, experts)

def run_single(image_path: str, mllm_type: str = "mock", mode: str = MOCK_MLLM_MODE,
               tools=None):
    """Run forensic analysis on a single image and print the result."""
    if not os.path.exists(image_path):
        print(f"Error: image not found — {image_path}")
        sys.exit(1)

    # Detect ground truth from path
    gt = "Real" if "/Real/" in image_path.replace("\\", "/") else "Fake"

    print(f"{'='*60}")
    print(f"Image:   {image_path}")
    print(f"GT:      {gt}")
    print(f"MLLM:    {mllm_type}")
    print(f"Mode:    {mode}")
    print(f"{'='*60}")

    fsm = build_pipeline(mllm_type, mode, tools=tools)
    result = fsm.run(image_path, ground_truth=gt)

    verdict = result["final_verdict"]
    print(f"\n  Verdict:    {verdict.get('verdict', 'N/A')}")
    print(f"  Confidence: {verdict.get('confidence', 'N/A'):.4f}")
    print(f"  Steps:      {result['total_steps']}")
    print(f"  Halting:    {result['halting_reason']}")
    print(f"  Evidence:   {len(result['evidence_chain'])} expert(s) called")
    for ev in result["evidence_chain"]:
        print(f"    - {ev['source']}: strength={ev['strength']:.4f} → {ev['support']}")
    print(f"  SFT data:   {result['sft_data_path']}")
    print()

    return result


def run_batch(category: str, mllm_type: str = "mock", mode: str = MOCK_MLLM_MODE,
              max_images: int = None, tools=None):
    """Run forensic analysis on a batch of images."""
    if category == "Real":
        image_dir = REAL_DIR
        gt = "Real"
    elif category in GENIMAGE_SUBDIRS:
        image_dir = os.path.join(GENIMAGE_TEST_DIR, category)
        gt = "Fake"
    elif category == "all":
        results = []
        # Real
        results.extend(run_batch("Real", mllm_type, mode, max_images, tools))
        # Each GenImage subdir
        for sub in GENIMAGE_SUBDIRS:
            results.extend(run_batch(sub, mllm_type, mode, max_images, tools))
        return results
    else:
        print(f"Unknown category: {category}")
        print(f"Valid options: Real, all, {', '.join(GENIMAGE_SUBDIRS)}")
        sys.exit(1)

    if not os.path.isdir(image_dir):
        print(f"Error: directory not found — {image_dir}")
        sys.exit(1)

    images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if max_images:
        images = images[:max_images]

    print(f"\nBatch: {category} ({len(images)} images, mode={mode})\n")

    results = []
    correct = 0
    for i, fname in enumerate(images):
        path = os.path.join(image_dir, fname)
        print(f"[{i+1}/{len(images)}] {fname} ... ", end="", flush=True)
        try:
            result = run_single(path, mllm_type, mode, tools=tools)
            results.append(result)
            v = result["final_verdict"].get("verdict", "")
            if (gt == "Real" and v == "Real") or (gt == "Fake" and v == "Fake"):
                correct += 1
                print("OK")
            else:
                print(f"WRONG (predicted {v})")
        except Exception as e:
            print(f"ERROR: {e}")

    accuracy = correct / len(images) if images else 0
    print(f"\n  Category: {category} | Accuracy: {correct}/{len(images)} = {accuracy:.2%}\n")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Active Forensic Agent — AI-generated image detection pipeline"
    )
    parser.add_argument(
        "--image", "-i", type=str,
        help="Path to a single image file for forensic analysis.",
    )
    parser.add_argument(
        "--mllm", type=str, default="mock", choices=["mock", "qwen", "qwen25"],
        help=("MLLM backend: 'mock' (template, CPU), 'qwen' = Qwen3-VL-8B-Instruct"
              "（当前，需 GPU）, 'qwen25' = Qwen2.5-VL-7B（仅历史复算）。"),
    )
    parser.add_argument(
        "--batch", "-b", type=str,
        help="Run batch analysis: 'Real', 'all', or a GenImage subdir name.",
    )
    parser.add_argument(
        "--mode", "-m", type=str, default=MOCK_MLLM_MODE,
        choices=MockMLLMClient.MODES,
        help=f"Mock MLLM behaviour mode (default: {MOCK_MLLM_MODE})",
    )
    parser.add_argument(
        "--max", "-n", type=int, default=None,
        help="Max images per batch category.",
    )
    parser.add_argument(
        "--tools", type=str, default=None,
        help=("Comma-separated tool policy, e.g. 'probe,trufor' for the minimal "
              "joint-task loop. Omit for the default probe,npr,univfd toolbox."),
    )

    args = parser.parse_args()
    tools = ([name.strip() for name in args.tools.split(",") if name.strip()]
             if args.tools else None)

    if args.image:
        run_single(args.image, args.mllm, args.mode, tools=tools)
    elif args.batch:
        run_batch(args.batch, args.mllm, args.mode, args.max, tools=tools)
    else:
        parser.print_help()
        print("\nExample: python main.py --image dataset/Real/xxx.jpg")
        print("         python main.py --batch Midjourney --max 5")


if __name__ == "__main__":
    main()
