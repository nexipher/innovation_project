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

from config import (
    EVIDENCE_ARTIFACT_DIR,
    MAX_EXPERT_CALLS,
    MAX_MODEL_TURNS,
    PROJECT_ROOT,
    SYSTEM_PROMPT,
    TURN_COST_WEIGHT,
)
from utils.image_utils import ImageUtils
from utils.coordinate_transformer import CoordinateTransformer
from utils.evidence_consistency import EvidenceConsistencyChecker
from utils.parser import Parser
from utils.reliability import ReliabilityTable
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

        # G2 §4.9: empirical reliability table (None before calibration exists)
        self._reliability_table = ReliabilityTable.load()

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

        # 4. Main loop — observable counters (plan.md §4.8 G1)
        model_turn_count = 0
        expert_call_count = 0
        suppressed_duplicate_count = 0
        seen_evidence_ids: set = set()
        evidence_chain: List[dict] = []
        final_verdict: Optional[dict] = None
        halting_reason: str = ""

        # Outer safety net: the halting budget fires first; the +1 allows the
        # forced-conclusion turn that budget halting triggers.
        while model_turn_count < MAX_MODEL_TURNS + 1:
            # 4a. Get MLLM response
            raw_output = self._mllm.generate(image_path, conversation)
            model_turn_count += 1

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

            if calls:
                # 4d. Execute each expert call
                for expert_name, rel_bbox in calls:
                    expert_call_count += 1

                    # Convert normalized [0,1000] → absolute pixel coordinates,
                    # keeping both spaces in the trace (plan.md §4.8 G1).
                    transform = CoordinateTransformer.transform(rel_bbox, w, h)
                    abs_bbox = transform["region_pixels"]

                    # Crop and run expert
                    patch = ImageUtils.crop_bbox(img, abs_bbox)
                    expert = self._experts.get(
                        self._expert_by_call.get(expert_name, "")
                    )
                    if expert is None:
                        continue

                    expert_result = expert.analyze(patch)

                    # G2 §4.9: attach empirical reliability / calibrated
                    # likelihood from the calibration table when available.
                    calibration = None
                    if self._reliability_table is not None:
                        calibration = self._reliability_table.lookup(
                            expert_result.source, expert_result.raw_metric
                        )

                    # Build Evidence Token
                    evidence_token = EvidenceTokenizer.tokenize(
                        expert_result, abs_bbox, (h, w),
                        region_normalized=transform["region_normalized_1000"],
                        reliability=calibration["reliability"] if calibration else None,
                        calibrated_likelihood=calibration["calibrated_likelihood"] if calibration else None,
                        condition_metadata=calibration["condition_metadata"] if calibration else None,
                        semantics_aligned=calibration["semantics_aligned"] if calibration else None,
                        applicability=calibration["applicability"] if calibration else None,
                        applicability_conditions=calibration["applicability_conditions"] if calibration else None,
                    )

                    # Deterministic semantic-consistency gate (plan.md §4.8 G1):
                    # direction words must agree with the measured strength.
                    EvidenceConsistencyChecker.enforce(evidence_token)

                    # Duplicate suppression (plan.md §4.8 G1): identical
                    # results must not enter the chain, the conversation or
                    # the halting statistics twice.
                    evidence_id = evidence_token["evidence_id"]
                    if evidence_id in seen_evidence_ids:
                        suppressed_duplicate_count += 1
                        continue
                    seen_evidence_ids.add(evidence_id)

                    # Persist the diagnostic region crop (G1) and the expert's
                    # visual artifacts (G2) so multi-turn image history can
                    # re-attach them.
                    artifact_rels = []
                    region_rel = self._save_artifact(patch, evidence_id, "region")
                    if region_rel:
                        evidence_token["diagnostic_region_image"] = region_rel
                        artifact_rels.append(region_rel)
                    try:
                        rendered = expert.render_artifacts(patch)
                    except Exception:
                        rendered = {}
                    for artifact_name, artifact_image in rendered.items():
                        rel = self._save_artifact(artifact_image, evidence_id, artifact_name)
                        if rel:
                            artifact_rels.append(rel)
                    if len(artifact_rels) > 1:
                        evidence_token["visual_artifacts"] = artifact_rels[1:]

                    # Record
                    evidence_chain.append(evidence_token)
                    self._logger.add_evidence(evidence_token)

                    # Inject into conversation as user message, attaching the
                    # region crop and artifacts for the real MLLM backend.
                    # Paths are stored project-relative so traces stay
                    # portable; the message builder resolves them at load time.
                    evidence_json = EvidenceTokenizer.to_json(evidence_token)
                    artifact_paths = artifact_rels or None
                    self._logger.add_conversation_turn(
                        "user", evidence_json, image_paths=artifact_paths
                    )
                    conversation.append({
                        "from": "user",
                        "value": evidence_json,
                        **({"image_paths": artifact_paths} if artifact_paths else {}),
                    })
            else:
                # No calls and no verdict — malformed output, inject correction
                valid, msg = Parser.validate_tag_structure(raw_output)
                if not valid:
                    correction = (
                        f"[System: 输出格式错误 — {msg} "
                        f"请严格按照 SOP 格式重新输出，包含 <call_*> 或 <verdict> 标签。]"
                    )
                    self._logger.add_system_message(correction)
                    conversation.append({"from": "user", "value": correction})

            # 4e. Check halting criteria
            should_halt, reason = HaltingChecker.check(
                model_turn_count, expert_call_count, evidence_chain, raw_output
            )
            if should_halt:
                halting_reason = reason

                # Budget / info gain: force model to produce verdict
                if reason in (
                    HaltingChecker.BUDGET_EXHAUSTED,
                    HaltingChecker.INFO_GAIN_CONVERGED,
                ):
                    budget_msg = (
                        "[System: 取证资源（Budget）已耗尽或信息增益收敛，"
                        "请立即基于已获取的全部证据撰写最终报告并输出 <verdict>。]"
                    )
                    final_output = self._request_conclusion(
                        image_path, conversation, budget_msg
                    )
                    model_turn_count += 1
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
                    final_output = self._request_conclusion(
                        image_path, conversation, conflict_msg
                    )
                    model_turn_count += 1
                    final_verdict = Parser.parse_verdict(final_output) or {
                        "verdict": "Uncertain",
                        "confidence": 0.45,
                        "report": "证据冲突，疑罪从无。",
                    }

                break

        # 5. If loop ended without verdict (emergency fallback)
        if final_verdict is None:
            halting_reason = halting_reason or HaltingChecker.BUDGET_EXHAUSTED
            final_verdict = {
                "verdict": "Uncertain",
                "confidence": 0.5,
                "report": "分析过程异常终止，无法做出确定结论。",
            }

        # 6. Finalise SFT data with observable counters
        counters = self._build_counters(
            model_turn_count, expert_call_count,
            suppressed_duplicate_count, evidence_chain,
        )
        self._logger.finalize_sft(
            final_verdict, model_turn_count, halting_reason, counters=counters
        )
        sft_path = self._logger.save_sft()

        return {
            "session_id": self._logger.session_id,
            "image_path": image_path,
            "ground_truth": ground_truth,
            "final_verdict": final_verdict,
            "total_steps": model_turn_count,       # legacy alias (deprecated)
            "model_turn_count": model_turn_count,
            "expert_call_count": expert_call_count,
            "unique_evidence_count": len(evidence_chain),
            "suppressed_duplicate_count": suppressed_duplicate_count,
            "weighted_cost": counters["weighted_cost"],
            "halting_reason": halting_reason,
            "evidence_chain": evidence_chain,
            "conversation": conversation,
            "sft_data_path": sft_path,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_artifact(self, image, evidence_id: str, name: str) -> Optional[str]:
        """
        Persist an evidence artifact (region crop or expert visualization).

        Returns the project-relative artifact path, or None when saving fails
        (artifact persistence must never break the analysis loop).
        """
        session_id = self._logger.session_id
        if not session_id or image is None or getattr(image, "size", 0) == 0:
            return None
        rel_path = os.path.join(
            "traces", "evidence", session_id, f"{name}_{evidence_id}.png"
        )
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        if ImageUtils.save_image(abs_path, image):
            return rel_path.replace(os.sep, "/")
        return None

    def _request_conclusion(
        self, image_path: str, conversation: List[Dict[str, str]], note: str
    ) -> str:
        """Inject a system note and ask the MLLM for its final verdict turn."""
        self._logger.add_system_message(note)
        conversation.append({"from": "user", "value": note})

        output = self._mllm.generate(image_path, conversation)
        self._logger.add_conversation_turn("gpt", output)
        conversation.append({"from": "gpt", "value": output})
        return output

    @staticmethod
    def _build_counters(
        model_turn_count: int,
        expert_call_count: int,
        suppressed_duplicate_count: int,
        evidence_chain: List[dict],
    ) -> dict:
        """Assemble the observable counter block recorded in the trace."""
        return {
            "model_turn_count": model_turn_count,
            "expert_call_count": expert_call_count,
            "unique_evidence_count": len(evidence_chain),
            "suppressed_duplicate_count": suppressed_duplicate_count,
            "weighted_cost": round(
                expert_call_count + TURN_COST_WEIGHT * model_turn_count, 4
            ),
        }
