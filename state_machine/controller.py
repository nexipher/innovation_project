"""
Forensic State Machine Controller — the central orchestrator.

Implements the core active-exploration loop:
  MLLM generates → parse tags → execute experts → inject Evidence Tokens
  → repeat until verdict or halting criteria trigger.

Pure Python while-loop; no external agent framework dependency.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

from config import SYSTEM_PROMPT, MAX_STEPS
from utils.image_utils import ImageUtils
from utils.coordinate_transformer import CoordinateTransformer
from utils.parser import Parser
from utils.logger import SessionLogger, log_operation
from state_machine.evidence_tokenizer import EvidenceTokenizer
from state_machine.halting import HaltingChecker


class ForensicStateMachine:
    """
    Central state machine that orchestrates the MLLM-Expert interaction loop.

    Usage:
        fsm = ForensicStateMachine(mllm_client, experts, logger)
        result = fsm.run("path/to/image.jpg")
    """

    # Expert registry — maps MLLM call tag names to expert instances
    EXPERT_KEY_MAP = {
        "freq": "frequency_expert",
        "noise": "noise_expert",
        "jpeg": "jpeg_expert",
    }

    # Expert registry — maps source_name to expert instances
    EXPERT_SOURCE_MAP = {
        "frequency_expert": "freq",
        "noise_expert": "noise",
        "jpeg_expert": "jpeg",
    }

    def __init__(
        self,
        mllm_client,
        experts: dict,
        logger: Optional[SessionLogger] = None,
    ):
        """
        Args:
            mllm_client: BaseMLLMClient instance (mock or real).
            experts: Dict mapping expert source_name → BaseExpert instance.
            logger: SessionLogger instance for SFT trace collection.
        """
        self._mllm = mllm_client
        self._experts = experts  # {"frequency_expert": ..., "noise_expert": ..., "jpeg_expert": ...}
        self._logger = logger or SessionLogger()

        # Build reverse lookup for expert dispatch
        self._expert_by_call: Dict[str, str] = {}
        for call_name, source_name in self.EXPERT_KEY_MAP.items():
            if source_name in self._experts:
                self._expert_by_call[call_name] = source_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        image_path: str,
        ground_truth: Optional[str] = None,
    ) -> dict:
        """
        Execute the full forensic analysis pipeline on a single image.

        Args:
            image_path: Absolute path to the image file.
            ground_truth: Optional label ("Real" or "Fake") for SFT metadata.

        Returns:
            Session result dict with keys:
              - session_id, image_path, ground_truth
              - final_verdict, total_steps, halting_reason
              - evidence_chain, conversation
              - sft_data_path
        """
        # 0. Reset MLLM state for new session
        self._mllm.reset()

        # 1. Load image
        img = ImageUtils.load_image(image_path)
        h, w = ImageUtils.get_dimensions(img)

        # 2. Initialise SFT session
        self._logger.init_sft_session(
            image_path=image_path,
            ground_truth=ground_truth,
            image_size=(h, w),
            mock_mode=getattr(self._mllm, "mode", ""),
        )

        # 3. Build initial conversation (system prompt guides SOP behaviour)
        conversation: List[Dict[str, str]] = []

        # 4. Main loop
        step = 0
        evidence_chain: List[dict] = []
        final_verdict: Optional[dict] = None
        halting_reason: str = ""

        while step < MAX_STEPS:
            # 4a. Get MLLM response
            raw_output = self._mllm.generate(image_path, conversation)

            # Log the assistant turn
            self._logger.add_conversation_turn("gpt", raw_output)
            conversation.append({"from": "gpt", "value": raw_output})

            # 4b. Check for verdict FIRST (model may conclude immediately)
            verdict = Parser.parse_verdict(raw_output)
            if verdict and "verdict" in verdict:
                final_verdict = verdict
                halting_reason = HaltingChecker.VERDICT_OUTPUT
                break

            # 4c. Extract expert calls
            calls = Parser.extract_all_calls(raw_output)

            if not calls:
                # No calls and no verdict — malformed output, inject correction
                valid, msg = Parser.validate_tag_structure(raw_output)
                if not valid:
                    correction = (
                        f"[System: 输出格式错误 — {msg} "
                        f"请严格按照 SOP 格式重新输出，包含 <call_*> 或 <verdict> 标签。]"
                    )
                    self._logger.add_system_message(correction)
                    conversation.append({"from": "user", "value": correction})
                step += 1
                continue

            # 4d. Execute each expert call
            for expert_name, rel_bbox in calls:
                # Convert relative → absolute pixel coordinates
                abs_bbox = CoordinateTransformer.relative_to_absolute(
                    rel_bbox, w, h
                )
                abs_bbox = CoordinateTransformer.clip_bbox(abs_bbox, w, h)

                # Crop and run expert
                patch = ImageUtils.crop_bbox(img, abs_bbox)
                expert = self._experts.get(
                    self._expert_by_call.get(expert_name, "")
                )
                if expert is None:
                    continue

                expert_result = expert.analyze(patch)

                # Build Evidence Token
                evidence_token = EvidenceTokenizer.tokenize(
                    expert_result, abs_bbox, (h, w)
                )

                # Record
                evidence_chain.append(evidence_token)
                self._logger.add_evidence(evidence_token)

                # Inject into conversation as user message
                evidence_json = EvidenceTokenizer.to_json(evidence_token)
                self._logger.add_conversation_turn("user", evidence_json)
                conversation.append({"from": "user", "value": evidence_json})

            step += 1

            # 4e. Check halting criteria
            should_halt, reason = HaltingChecker.check(
                step, evidence_chain, raw_output
            )
            if should_halt:
                halting_reason = reason

                # Max steps / info gain: force model to produce verdict
                if reason in (
                    HaltingChecker.MAX_STEPS_EXCEEDED,
                    HaltingChecker.INFO_GAIN_CONVERGED,
                ):
                    budget_msg = (
                        "[System: 取证资源（Budget）已耗尽或信息增益收敛，"
                        "请立即基于已获取的全部证据撰写最终报告并输出 <verdict>。]"
                    )
                    self._logger.add_system_message(budget_msg)
                    conversation.append({"from": "user", "value": budget_msg})

                    final_output = self._mllm.generate(image_path, conversation)
                    self._logger.add_conversation_turn("gpt", final_output)
                    conversation.append({"from": "gpt", "value": final_output})
                    final_verdict = Parser.parse_verdict(final_output) or {
                        "verdict": "Uncertain",
                        "confidence": 0.5,
                        "report": "取证资源耗尽，无法做出确定结论。",
                    }

                # Conflict: force reflection + uncertain verdict
                elif reason == HaltingChecker.EVIDENCE_CONFLICT:
                    conflict_msg = (
                        "[System: 法证证据出现强冲突（疑罪从无 — Conflict Halting），"
                        "请进行双向反思并输出 Uncertain 置信度校准结论。]"
                    )
                    self._logger.add_system_message(conflict_msg)
                    conversation.append({"from": "user", "value": conflict_msg})

                    final_output = self._mllm.generate(image_path, conversation)
                    self._logger.add_conversation_turn("gpt", final_output)
                    conversation.append({"from": "gpt", "value": final_output})
                    final_verdict = Parser.parse_verdict(final_output) or {
                        "verdict": "Uncertain",
                        "confidence": 0.45,
                        "report": "证据冲突，疑罪从无。",
                    }

                break

        # 5. If loop ended without verdict (emergency fallback)
        if final_verdict is None:
            halting_reason = halting_reason or "max_steps_exceeded"
            final_verdict = {
                "verdict": "Uncertain",
                "confidence": 0.5,
                "report": "分析过程异常终止，无法做出确定结论。",
            }

        # 6. Finalise SFT data
        self._logger.finalize_sft(final_verdict, step, halting_reason)
        sft_path = self._logger.save_sft()

        return {
            "session_id": self._logger.session_id,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "final_verdict": final_verdict,
            "total_steps": step,
            "halting_reason": halting_reason,
            "evidence_chain": evidence_chain,
            "conversation": conversation,
            "sft_data_path": sft_path,
        }
