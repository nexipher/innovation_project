"""
Forensic State Machine Controller — the central orchestrator.

Implements the core active-exploration loop:
  MLLM generates → parse tags → execute experts → inject Evidence Tokens
  → repeat until verdict or halting criteria trigger.

Pure Python while-loop; no external agent framework dependency.
"""

import json
import math
import os
from typing import Dict, List, Optional

import numpy as np

from config import (
    uses_reference_posterior,
    EVIDENCE_ARTIFACT_DIR,
    HALTING_POLICY,
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
from state_machine.evidence_rectifier import EvidenceRectifier
from utils.evidence_ledger import Ledger as EvidenceLedger, correction_note
from state_machine.evidence_tokenizer import EvidenceTokenizer
from state_machine.halting import HaltingChecker
from state_machine.halting_v2 import HaltingPolicyV2
from state_machine.halting_v3 import MODEL_CONFIDENCE_FALLBACK, HaltingPolicyV3


def _model_confidence(value) -> float:
    """
    The model's own confidence, as it wrote it.

    `value or 0.5` treats a legitimate 0.0 as missing — it did, and the
    repaired confidence was reported instead.  The check is on the field
    being absent, and the number is then bounded and checked for finiteness:
    a NaN, an infinity or a 7 from the model is a missing value, not a
    confidence, and is recorded as the documented fallback.
    """
    if value is None:
        return MODEL_CONFIDENCE_FALLBACK
    try:
        number = float(value)
    except (TypeError, ValueError):
        return MODEL_CONFIDENCE_FALLBACK
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        return MODEL_CONFIDENCE_FALLBACK
    return round(number, 4)


class ForensicStateMachine:
    """
    Central state machine that orchestrates the MLLM-Expert interaction loop.

    Usage:
        fsm = ForensicStateMachine(mllm_client, experts, logger)
        result = fsm.run("path/to/image.jpg")
    """

    # Expert registry — maps MLLM call tag names to registry keys.
    # G2-e (plan.md §4.9): the runtime ships the v2 frequency expert, and
    # its source name is distinct from v1's, so calibration lookups reach the
    # v2 entry rather than v1's disabled:no-signal one.  v1 is never
    # registered; ELA stays out too (its G2-b skill is compression history).
    EXPERT_KEY_MAP = {
        # G7 entry 3: PROBE is the main detector — C-group AUROC 0.986 against
        # the old toolkit's 0.764, and the fusion that would have combined them
        # showed no gain outside noise.  The three below stay as corroborating
        # clues: they are offered, never required.
        "probe": "probe_dino_expert",
        "freq": "frequency_expert_v2",
        "noise": "noise_expert",
        "jpeg": "jpeg_expert",
    }

    # Expert registry — maps source_name to MLLM call tag
    EXPERT_SOURCE_MAP = {
        "probe_dino_expert": "probe",
        "frequency_expert_v2": "freq",
        "frequency_expert": "freq",      # deprecated v1 (old traces only)
        "noise_expert": "noise",
        "jpeg_expert": "jpeg",
    }

    def __init__(
        self,
        mllm_client,
        experts: dict,
        logger: Optional[SessionLogger] = None,
        evidence_injection: str = "text+image",
        allow_exploration: bool = True,
        condition: Optional[str] = None,
    ):
        """
        Args:
            mllm_client: BaseMLLMClient instance (mock or real).
            experts: Dict mapping expert source_name → BaseExpert instance.
            logger: SessionLogger instance for SFT trace collection.
            evidence_injection: How executed evidence reaches the MLLM
                conversation (G2-d experiment axis):
                  - "text+image" (default, G2-c): Evidence Token JSON plus the
                    region crop and expert artifacts;
                  - "text" (G1 behaviour): Evidence Token JSON only;
                  - "image": region crop and artifacts, with the text replaced
                    by a neutral marker (no numbers are shown to the model);
                  - "none": nothing is surfaced (RGB baseline); calls are still
                    executed and recorded for audit.
            condition: The image's format cell ("png" / "jpeg_q95" / ...) when
                the caller knows it — the calibration table's per-format
                separations are then used as the token weights instead of the
                pooled value (G7 A2).  None keeps the pooled weight.
            allow_exploration: False when the session's protocol forbids tool
                calls (the no-tool baseline).  The halting policy then judges
                the model's own verdict instead of demanding evidence that
                cannot arrive — otherwise the arm spins to the turn budget and
                abstains on everything, which measures the policy rather than
                the model.
        """
        self._condition = condition
        if evidence_injection not in ("text+image", "text", "image", "none"):
            raise ValueError(f"unknown evidence_injection: {evidence_injection}")

        self._mllm = mllm_client
        self._experts = experts  # {"frequency_expert": ..., "noise_expert": ..., "jpeg_expert": ...}
        self._logger = logger or SessionLogger()
        self._evidence_injection = evidence_injection
        self._allow_exploration = allow_exploration

        # 参考后验只属于 v1/v2 复算；v3 没有读者，就不加载校准表——"关闭"必须是
        # 真的不读。这里按**运行中的策略**求值（策略可在运行时切换），不是 import 时常量。
        self._uses_reference = uses_reference_posterior(HALTING_POLICY)
        self._reliability_table = (ReliabilityTable.load()
                                   if self._uses_reference else None)

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
        skipped_duplicate_calls = 0
        evidence_check: dict = {"status": "not_run", "findings": []}
        skipped_sources: List[str] = []
        measured_sources: Dict[str, str] = {}     # source_name → evidence_id
        seen_evidence_ids: set = set()
        evidence_chain: List[dict] = []
        final_verdict: Optional[dict] = None
        halting_reason: str = ""
        called_experts: set = set()
        policy_decision = None
        turns_without_new_evidence = 0
        # §4.17 R5 (v3): loop guards measured on actions, and the advisory
        # note the reference is allowed to send exactly once.
        previous_action = None
        repeated_action_turns = 0
        invalid_action_turns = 0
        format_failure_turns = 0
        advisory_sent = False
        # §4.17 R9: turns refused for carrying more than one action.
        protocol_violations = 0

        # Outer safety net: the halting budget fires first; the +1 allows the
        # forced-conclusion turn that budget halting triggers.
        while model_turn_count < MAX_MODEL_TURNS + 1:
            # 4a. Get MLLM response
            raw_output = self._mllm.generate(image_path, conversation)
            model_turn_count += 1

            # Log the assistant turn
            self._logger.add_conversation_turn("gpt", raw_output)
            conversation.append({"from": "gpt", "value": raw_output})
            evidence_before_turn = len(evidence_chain)

            # 4b. A verdict tag is a *candidate*: v2 weighs it against the
            # posterior, v1 halts on it immediately (kept for replay).
            candidate_verdict = Parser.parse_verdict(raw_output)
            if candidate_verdict and "verdict" in candidate_verdict and HALTING_POLICY == "v1":
                # §4.17 R4: even the v1 shortcut is annotated by the ledger.
                # v1 is frozen for replay, so the check does not correct here
                # (a correction turn would change the protocol it records);
                # a v1 session that fabricates is marked failed and the
                # conversion gate refuses it.
                v1_ledger = EvidenceLedger.from_session(
                    evidence_chain, skipped_sources=skipped_sources,
                    served=list(self._experts),
                    system_texts=self._system_texts(conversation),
                )
                v1_findings = v1_ledger.findings(raw_output)
                v1_defects = [f for f in v1_findings
                              if f.get("severity", "defect") == "defect"]
                candidate_verdict["evidence_check"] = {
                    "status": "pass" if not v1_defects else "failed",
                    "findings": v1_defects,
                    "candidates": [f for f in v1_findings if f not in v1_defects],
                    "entry": "v1_candidate",
                    "note": "v1 协议冻结：只标注不纠正",
                    "caveat": "未命中不等于已证明忠实：检查只覆盖已定义的模式",
                }
                candidate_verdict.setdefault("verdict_status", "model_concluded")
                candidate_verdict["conclusion"] = {
                    "source": "v1_candidate",
                    "calls": [name for name, _ in Parser.executable_calls(raw_output)],
                    "calls_executed": False,
                    "verdict_parse": "ok",
                }
                final_verdict = candidate_verdict
                halting_reason = HaltingChecker.VERDICT_OUTPUT
                break

            # 4c. Extract expert calls — one action per turn (§4.17 R9).
            # A turn that mixes a call with a conclusion, or asks for two
            # tools at once, is answered with a protocol correction and
            # **nothing is executed**: the executor never picks a half for
            # the model, and the trace records that the turn was refused.
            action = Parser.protocol_action(raw_output)
            if action["violation"]:
                protocol_violations += 1
                protocol = (
                    "[System: 协议要求每轮只有一个动作 —— 要么一次工具调用"
                    "（一个 <call_x> 标签），要么一次结案（一个 <verdict> 标签）。"
                    "本轮同时出现了多个动作，因此都未执行。"
                    "请重新输出：只保留你要做的那个动作。]"
                )
                self._logger.add_system_message(protocol)
                conversation.append({"from": "user", "value": protocol})
                calls, candidate_verdict = [], None
            else:
                calls = action["calls"]
            turn_calls_executed = True
            turn_signature_parts: list = []

            if calls:
                # 4d. Execute each expert call
                turn_calls_executed = True   # did every call in this turn run?
                for expert_name, rel_bbox in calls:
                    # Convert normalized [0,1000] → absolute pixel coordinates,
                    # keeping both spaces in the trace (plan.md §4.8 G1).
                    # §4.17 R9: a call without a bbox is legal — every expert
                    # measures the whole image, and the region was only ever a
                    # diagnostic note.  Nothing is invented here: the trace
                    # records that no region was requested.
                    if rel_bbox is None:
                        transform = {"region_pixels": [0, 0, h, w],
                                     "region_normalized_1000": None}
                        abs_bbox = [0, 0, h, w]
                        patch = img
                    else:
                        transform = CoordinateTransformer.transform(rel_bbox, w, h)
                        abs_bbox = transform["region_pixels"]
                        patch = ImageUtils.crop_bbox(img, abs_bbox)

                    source_name = self._expert_by_call.get(expert_name, "")
                    expert = self._experts.get(source_name)
                    if expert is None:
                        continue

                    # G7 entry 3 (measured in the acceptance run): every expert
                    # measures the whole image deterministically, so a second
                    # call to the same expert in one session *cannot* return
                    # anything new.  The model does it anyway — 48.6% of all
                    # executions in the 120-session batch were repeats — and
                    # running the expert first and deduplicating on evidence_id
                    # afterwards pays the GPU time and the call budget before
                    # noticing.  So the check moves in front of the measurement:
                    # an expensive call that cannot add information is not made,
                    # and it is not charged to the budget either (the budget
                    # buys measurements, not repetitions).
                    if source_name in measured_sources:
                        skipped_duplicate_calls += 1
                        turn_calls_executed = False
                        # A repeat request is an action that produced nothing;
                        # the loop guard must see it as one (the bbox differs,
                        # the measurement does not).
                        turn_signature_parts.append(("repeat", source_name))
                        if source_name not in skipped_sources:
                            skipped_sources.append(source_name)
                        notice = (
                            f"[System: {source_name} 已在本会话对整图测量过"
                            f"（{measured_sources[source_name]}），整图测量与区域无关，"
                            f"重复调用不会产生新的测量，已跳过执行。请基于现有证据继续。]"
                        )
                        self._logger.add_conversation_turn("user", notice)
                        conversation.append({"from": "user", "value": notice})
                        continue

                    expert_call_count += 1
                    turn_signature_parts.append(("call", source_name))

                    # G3-a: the measurement is taken on the whole image.  The
                    # reliability table is calibrated on full images (n=700),
                    # while a crop has a different metric distribution —
                    # measuring the crop and then reading the full-image table
                    # binned the evidence against the wrong population (noise
                    # crop median 1.262 vs 2.529 full).  The model's bbox keeps
                    # its role as the diagnostic and visualisation region.
                    expert_result = expert.analyze(img)

                    # G2 §4.9: attach empirical reliability / calibrated
                    # likelihood from the calibration table when available.
                    calibration = None
                    if self._reliability_table is not None:
                        calibration = self._reliability_table.lookup(
                            expert_result.source, expert_result.raw_metric,
                            condition=self._condition,
                        )

                    # Build Evidence Token
                    evidence_token = EvidenceTokenizer.tokenize(
                        expert_result, abs_bbox, (h, w),
                        region_normalized=transform["region_normalized_1000"],
                        measurement_scope="global",
                        region_area_ratio=(
                            (abs_bbox[2] - abs_bbox[0]) * (abs_bbox[3] - abs_bbox[1])
                            / float(h * w)
                        ) if h and w else None,
                        reliability=calibration["reliability"] if calibration else None,
                        calibrated_likelihood=calibration["calibrated_likelihood"] if calibration else None,
                        condition_metadata=calibration["condition_metadata"] if calibration else None,
                        semantics_aligned=calibration["semantics_aligned"] if calibration else None,
                        applicability=calibration["applicability"] if calibration else None,
                        applicability_conditions=calibration["applicability_conditions"] if calibration else None,
                        calibration=self._calibration_block(calibration),
                    )

                    # G3-b: settle the token's direction before anything reads
                    # it — the calibrated likelihood is the authority, the
                    # expert's claim and its prose are brought into line.
                    # v3 的方向权威是分数语义登记表；v2 复算仍走校准表
                    # （replay 必须复现它产出的批次，方向不能换来源）。
                    EvidenceRectifier.rectify(
                        evidence_token, prefer_registry=not self._uses_reference)

                    # Deterministic semantic-consistency gate (plan.md §4.8 G1):
                    # for a rectified token this is now a verification that
                    # should pass; uncalibrated tokens still get the old guard.
                    EvidenceConsistencyChecker.enforce(evidence_token)

                    # Duplicate suppression (plan.md §4.8 G1): identical
                    # results must not enter the chain, the conversation or
                    # the halting statistics twice.
                    evidence_id = evidence_token["evidence_id"]
                    if evidence_id in seen_evidence_ids:
                        suppressed_duplicate_count += 1
                        # The call still has to be answered.  Swallowing it
                        # silently left the model's turn dangling in the
                        # session log and taught it that a call may be
                        # followed by a conclusion with nothing in between
                        # (29 samples in the 2026-10-11 review).
                        notice = (
                            f"[System: 本次调用返回的证据与已有的 {evidence_id} 完全"
                            f"相同，已按去重规则忽略，未产生新的测量。请基于现有证据"
                            f"继续，不要重复调用同一工具与同一区域。]"
                        )
                        self._logger.add_conversation_turn("user", notice)
                        conversation.append({"from": "user", "value": notice})
                        continue
                    seen_evidence_ids.add(evidence_id)
                    called_experts.add(expert_result.source)
                    measured_sources[expert_result.source] = evidence_id
                    turns_without_new_evidence = -1  # counted at the turn's end

                    # Persist the diagnostic region crop (G1) and the expert's
                    # visual artifacts (G2) when the current condition will
                    # actually show them. Paths are project-relative so traces
                    # stay portable; the message builder resolves them on load.
                    attach_images = self._evidence_injection in ("image", "text+image")
                    artifact_rels = []
                    if attach_images:
                        region_rel = self._save_artifact(patch, evidence_id, "region")
                        if region_rel:
                            evidence_token["diagnostic_region_image"] = region_rel
                            artifact_rels.append(region_rel)
                        try:
                            # Rendered from the same input the metric was
                            # measured on, so the picture and the number
                            # describe one thing (G3-a).
                            rendered = expert.render_artifacts(img)
                        except Exception:
                            rendered = {}
                        for artifact_name, artifact_image in rendered.items():
                            rel = self._save_artifact(artifact_image, evidence_id, artifact_name)
                            if rel:
                                artifact_rels.append(rel)
                        if len(artifact_rels) > 1:
                            evidence_token["visual_artifacts"] = artifact_rels[1:]

                    # Record (always — the chain is the audit source of truth)
                    evidence_chain.append(evidence_token)
                    self._logger.add_evidence(evidence_token)

                    # Surface to the model according to the injection mode
                    # (G2-d experiment axis: text channel vs visual channel).
                    if self._evidence_injection == "none":
                        pass
                    elif self._evidence_injection == "text":
                        evidence_json = EvidenceTokenizer.to_json(evidence_token)
                        self._logger.add_conversation_turn("user", evidence_json)
                        conversation.append({"from": "user", "value": evidence_json})
                    elif self._evidence_injection == "image":
                        # G3-a: the metric is measured on the whole image, so
                        # the artifacts are whole-image products.  Calling them
                        # "the analysis of region [bbox]" would have the model
                        # read a global spectrum as a local finding — the very
                        # confusion this condition exists to test without.
                        marker = (
                            f"[诊断证据图已附：{expert_result.source} 在**整幅图像**上"
                            f"计算法证指标；所附产物图均为整图产物，其中区域图对应你提出的"
                            f"关注区域 {abs_bbox}（仅为诊断关注点，不限定测量范围）。"
                            f"请结合图像自行判断。]"
                        )
                        self._logger.add_conversation_turn(
                            "user", marker, image_paths=artifact_rels
                        )
                        conversation.append({
                            "from": "user", "value": marker,
                            "image_paths": artifact_rels,
                        })
                    else:  # "text+image" (default)
                        evidence_json = EvidenceTokenizer.to_json(evidence_token)
                        self._logger.add_conversation_turn(
                            "user", evidence_json, image_paths=artifact_rels or None
                        )
                        conversation.append({
                            "from": "user",
                            "value": evidence_json,
                            **({"image_paths": artifact_rels} if artifact_rels else {}),
                        })
                        # §4.17: the state, restated after every measurement.
                        # The token stays above; this line is what stops the
                        # model from going back for a measurement it already
                        # has (41 of 56 acceptance sessions did).
                        state_note = self._state_note(
                            evidence_chain, list(self._experts), measured_sources)
                        self._logger.add_conversation_turn("user", state_note)
                        conversation.append({"from": "user", "value": state_note})
            else:
                # No calls and no verdict — malformed output, inject the
                # correction that says what is actually wrong.
                valid, msg = Parser.validate_tag_structure(raw_output)
                if not valid:
                    correction = self._format_correction(raw_output, msg)
                    self._logger.add_system_message(correction)
                    conversation.append({"from": "user", "value": correction})

            # 4e. Ask the halting policy what to do next
            if HALTING_POLICY in ("v2", "v3"):
                turns_without_new_evidence = (
                    0 if len(evidence_chain) > evidence_before_turn
                    else max(0, turns_without_new_evidence) + 1
                )
                # §4.17 R5: v3's loop guards are about *actions* — the same
                # action again, or a turn that never became one — not about
                # "no new evidence", which a legal first call also produces.
                if calls:
                    action_signature = tuple(turn_signature_parts) or ("call",)
                elif candidate_verdict:
                    action_signature = ("verdict",)
                else:
                    action_signature = ()
                    format_failure_turns += 1
                # A repeat request is a no-progress action whether or not it
                # repeats the previous turn verbatim: the measurement it asks
                # for is already in the conversation.
                no_progress = any(part[0] == "repeat"
                                  for part in turn_signature_parts)
                if action_signature and (no_progress or action_signature == previous_action):
                    repeated_action_turns += 1
                elif action_signature:
                    repeated_action_turns = 0
                invalid_action_turns = 0 if action_signature else invalid_action_turns + 1
                previous_action = action_signature

                if HALTING_POLICY == "v3":
                    decision = HaltingPolicyV3.decide(
                        compute_reference=self._uses_reference,
                        model_turns=model_turn_count,
                        expert_calls=expert_call_count,
                        evidence_chain=evidence_chain,
                        candidate_verdict=candidate_verdict,
                        available_experts=self._expert_weights(),
                        called_experts=called_experts,
                        repeated_actions=repeated_action_turns,
                        invalid_actions=invalid_action_turns,
                        tools_available=self._allow_exploration,
                    )
                else:
                    decision = HaltingPolicyV2.decide(
                        model_turns=model_turn_count,
                        expert_calls=expert_call_count,
                        evidence_chain=evidence_chain,
                        candidate_verdict=candidate_verdict,
                        available_experts=self._expert_weights(),
                        called_experts=called_experts,
                        turns_without_new_evidence=turns_without_new_evidence,
                        tools_available=self._allow_exploration,
                    )
                policy_decision = decision
                if decision.action == "continue":
                    if HALTING_POLICY == "v3":
                        if candidate_verdict:
                            # A <verdict> tag with a label that is not
                            # Real/Fake/Uncertain: a protocol problem, and it
                            # is answered as one — never by quietly choosing a
                            # label the model did not write.
                            protocol = (
                                "[System: <verdict> 中的 verdict 必须是 Real / Fake / "
                                "Uncertain 之一。请按 SOP 重新输出你的结案。]"
                            )
                            self._logger.add_system_message(protocol)
                            conversation.append({"from": "user", "value": protocol})
                        elif decision.advisory and not advisory_sent:
                            # The reference's estimate, marked as advice.  It
                            # never forbids a call: v3 executes legal calls
                            # whether or not the estimate is positive.
                            advisory_sent = True
                            note = (
                                "[System: 策略的参考估计认为继续调用收益有限，"
                                "但这只是参考，不影响你也无权阻止你的取证判断；"
                                "你可以继续调用未用过的工具，或给出结案。]"
                            )
                            self._logger.add_system_message(note)
                            conversation.append({"from": "user", "value": note})
                    elif candidate_verdict:
                        # A verdict tag was overridden: say so, or the model
                        # will simply repeat it until the budget runs out.
                        override = (
                            f"[System: 你的结论与当前校准后验不一致"
                            f"（P(Fake)={decision.posterior:.2f}，"
                            f"冲突度={decision.conflict_score:.2f}），"
                            f"暂不能结案。请继续取证：建议调用 "
                            f"{decision.next_expert or '下一个未使用的专家'}，"
                            f"或说明为何现有证据足以支持你的结论。]"
                        )
                        self._logger.add_system_message(override)
                        conversation.append({"from": "user", "value": override})
                    continue
                halting_reason = decision.primary_reason
                # v2: the policy owns the label.  v3: the model does — the
                # policy only decided that the session ends (§4.17 R5).
                if HALTING_POLICY == "v3":
                    accepted = (candidate_verdict or {}).get("verdict") in (
                        HaltingPolicyV3.LEGAL_LABELS) and bool(
                        (candidate_verdict or {}).get("report"))
                else:
                    accepted = decision.primary_reason in (
                        HaltingPolicyV2.CANDIDATE_ACCEPTED,
                        HaltingPolicyV2.CANDIDATE_WITHOUT_TOOLS,
                    )
                # §4.17 R3/R4: the ledger is the only authority on what was
                # measured, and **every** conclusion this pipeline adopts goes
                # through it — the accepted-candidate shortcut used to skip the
                # check entirely, so a fabrication inside an accepted candidate
                # would have been rendered into training data unexamined.  One
                # correction is allowed; the original and the correction are
                # both recorded, never silently dropped.
                ledger = EvidenceLedger.from_session(
                    evidence_chain, skipped_sources=skipped_sources,
                    served=list(self._experts),
                    system_texts=self._system_texts(conversation),
                )
                if accepted and (candidate_verdict or {}).get("report"):
                    final_output = raw_output
                    adopted_from = "candidate"
                    adopted_calls = [name for name, _ in calls]
                    adopted_calls_executed = turn_calls_executed
                else:
                    note = self._conclusion_note(
                        decision, ledger, turns_without_new_evidence)
                    final_output = self._request_conclusion(
                        image_path, conversation, note
                    )
                    model_turn_count += 1
                    adopted_from = "closing_turn"
                    # A closing turn's calls are never executed (the controller
                    # does not run the measurement engine from this entry), so
                    # any call written here is recorded as unexecuted.
                    adopted_calls = [name for name, _ in Parser.executable_calls(final_output)]
                    adopted_calls_executed = False
                findings = ledger.findings(final_output)
                # §4.17 R5: only *defects* are corrected automatically.  A
                # candidate finding (a direction read out of a near-chance
                # value, a number that only exists in an expert's prose) is a
                # judgement call: it is recorded for a person and never spent
                # on the single correction the session gets.
                defects = [f for f in findings if f.get("severity", "defect") == "defect"]
                candidates = [f for f in findings
                              if f.get("severity", "defect") != "defect"]
                evidence_check = {
                    "status": "pass", "findings": [], "candidates": findings,
                    "entry": adopted_from,
                    "caveat": "未命中不等于已证明忠实：检查只覆盖已定义的模式",
                }
                if defects:
                    correction = correction_note(ledger, defects)
                    corrected = self._request_conclusion(
                        image_path, conversation, correction)
                    model_turn_count += 1
                    remaining = ledger.findings(corrected)
                    remaining_defects = [f for f in remaining
                                         if f.get("severity", "defect") == "defect"]
                    evidence_check = {
                        "status": "pass_after_correction" if not remaining_defects else "failed",
                        "findings": defects,
                        "after_correction": remaining_defects,
                        "candidates": candidates + [
                            f for f in remaining
                            if f.get("severity", "defect") != "defect"],
                        "correction_note": correction,
                        "original_closing": final_output[-4000:],
                        "entry": adopted_from,
                        "caveat": "未命中不等于已证明忠实：检查只覆盖已定义的模式",
                    }
                    final_output = corrected
                    # Calls written into a correction turn are never executed:
                    # a training sample must not show a call as part of the
                    # answer that produced a verdict.
                    adopted_from = "correction"
                    adopted_calls = [name for name, _ in Parser.executable_calls(final_output)]
                    adopted_calls_executed = False
                model_candidate = Parser.parse_verdict(final_output) or {}
                conclusion = {
                    "source": adopted_from,
                    "calls": adopted_calls,
                    "calls_executed": adopted_calls_executed,
                    "verdict_parse": "ok" if model_candidate.get("verdict") else "failed",
                }
                # §4.17 R5: in v3 the label is whatever the adopted conclusion
                # says — `Real`, `Fake`, `Uncertain` or nothing parseable.
                # The posterior is recorded beside it and never becomes it.
                # A session with no parseable conclusion is *not* an
                # abstention: it is a failed conclusion, and the two are
                # counted apart (`verdict_status`).
                if HALTING_POLICY == "v3":
                    proposed = model_candidate.get("verdict")
                    if proposed in HaltingPolicyV3.LEGAL_LABELS:
                        final_label = proposed
                        verdict_status = "model_concluded"
                    else:
                        final_label = "Uncertain"
                        verdict_status = "no_parseable_conclusion"
                    final_confidence = _model_confidence(model_candidate.get("confidence"))
                else:
                    final_label = decision.verdict
                    final_confidence = decision.confidence
                    verdict_status = "policy_label"
                final_verdict = {
                    "verdict": final_label,
                    # Was this label the model's own conclusion, or the
                    # fallback for a conclusion that never parsed?  Merging
                    # the two would inflate the abstention rate.
                    "verdict_status": verdict_status,
                    "confidence": final_confidence,
                    "report": model_candidate.get("report") or (
                        "取证后仍未形成可判定方向，按疑罪从无输出 Uncertain。"
                        if verdict_status == "no_parseable_conclusion"
                        else "基于校准后验与已获取证据的判定。"
                    ),
                    # 参考后验与预计收益只属于 v1/v2 复算；v3 里这些字段**不出现**，
                    # 而不是写成 0.5/0.0 那种"看起来像算过"的缺省值。
                    **({
                        "posterior": round(decision.posterior, 4),
                        "tool_utilities": dict(decision.tool_utilities or {}),
                        "conflict_score": round(decision.conflict_score, 4),
                        "redundancy_discounts": HaltingPolicyV2.discount_report(evidence_chain),
                    } if self._uses_reference else {}),
                    "policy_reason": decision.primary_reason,
                    "policy_reasons": list(decision.all_reasons),
                    "model_candidate": model_candidate.get("verdict"),
                    "evidence_check": evidence_check,
                    # §4.17 R4: check-pass / format-valid / conclusion-faithful
                    # / trainable are four different things.  This block is the
                    # first three; `trainable` is the conversion gate's call.
                    "conclusion": conclusion,
                    # §4.17 R5 的参考块只在 v1/v2 复算里出现；v3 没有参考，
                    # 就不留一个"空参考"字段让人以为它算了但没算出来。
                    **({
                        "reference": {
                            "recommendation": getattr(decision, "reference_label", None),
                            "advisory": list(getattr(decision, "advisory", []) or []),
                            "note": "参考后验与建议不构成标签；标签一律取自模型结案",
                        },
                    } if self._uses_reference else {}),
                }
                break

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
                # the loop never produced a conclusion: not an abstention
                "verdict_status": "no_parseable_conclusion",
                "confidence": 0.5,
                "report": "分析过程异常终止，无法做出确定结论。",
            }

        # 6. Finalise SFT data with observable counters
        v3_signals = None
        if HALTING_POLICY == "v3":
            v3_signals = {
                "repeated_action_turns": repeated_action_turns,
                "invalid_action_turns": invalid_action_turns,
                "format_failure_turns": format_failure_turns,
                "model_label": (final_verdict or {}).get("model_candidate"),
            }
            # 参考派生信号只在真算了参考时记录（阶段 4.2）：不写 None，
            # 免得读者以为"算了但没算出来"。
            if self._uses_reference:
                reference = (final_verdict or {}).get("reference") or {}
                v3_signals["advisory_notes"] = list(reference.get("advisory") or [])
                v3_signals["reference_recommendation"] = reference.get("recommendation")
        counters = self._build_counters(
            model_turn_count, expert_call_count,
            suppressed_duplicate_count, evidence_chain,
            skipped_duplicate_calls=skipped_duplicate_calls,
            policy_decision=policy_decision,
            available_experts=list(self._experts),
            evidence_check=evidence_check,
            conclusion=(final_verdict or {}).get("conclusion"),
            v3_signals=v3_signals,
            protocol_violations=protocol_violations,
            condition=self._condition,
            verdict_status=(final_verdict or {}).get("verdict_status"),
            verdict_label=(final_verdict or {}).get("verdict"),
            reference_fields=self._uses_reference,
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
            "skipped_duplicate_calls": skipped_duplicate_calls,
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

    @staticmethod
    def _calibration_block(calibration: Optional[dict]) -> Optional[dict]:
        """
        Which mapping produced this token's probability, and where it was fitted.

        The numbers travel with their origin: a reader of the trace (or of the
        SFT sample built from it) must be able to tell an A-group mapping from
        something fitted on the group it is being judged on.  None when the
        lookup returned nothing — absent means "no mapping", not "unknown".
        """
        if not calibration:
            return None
        provenance = calibration.get("provenance") or {}
        metadata = calibration.get("condition_metadata") or {}
        return {
            "table": "calibration/reliability_table.json",
            "fitted_on": provenance.get("fitted_on"),
            "mapping": provenance.get("mapping"),
            "sources": provenance.get("sources"),
            "samples": provenance.get("samples"),
            "split_hash": provenance.get("split_hash"),
            "weights_sha256": provenance.get("weights_sha256"),
            # The mapping stays a pooled quantile lookup; only the *weight* is
            # per format.  Saying that here keeps "per-format calibration"
            # from being read into the probabilities (§4.17 acceptance).
            "mapping_scope": "pooled_quantile_bins",
            "weight_basis": metadata.get("weight_basis"),
            "weight": metadata.get("weight"),
            "condition": metadata.get("condition"),
        }

    def _expert_weights(self) -> Dict[str, float]:
        """
        Measured decision weight per registered expert (G3-c).

        The calibration table's polarity-corrected separation is what the
        expert can still tell apart after every confound G2-b could remove;
        experts without a measurement get no weight, so the policy will not
        pay for calling them.
        """
        if not self._uses_reference:
            # v3 不预计收益，也就不必读校准表：没有读者。
            return {}
        separations: Dict[str, Optional[float]] = {}
        for source_name in self._experts:
            entry = None
            if self._reliability_table is not None:
                entry = self._reliability_table.expert_entry(source_name)
            if entry is None:
                separations[source_name] = None
                continue
            # The format's own measured separation when it is known (A2).
            separations[source_name] = self._reliability_table.weight_of(
                entry, self._condition)[0]
        return HaltingPolicyV2.expert_weights(separations)

    @staticmethod
    def _format_correction(text: str, message: str) -> str:
        """
        A correction the model can act on (§4.17).

        "请包含 <call_*> 或 <verdict> 标签" is useless when the model *did*
        write a call tag — inside <planning>, where it is read as prose.  One
        BigGAN session did that three times and died as a format failure with
        a message telling it to do what it had already done.  When the tags
        are buried, the correction quotes the exact line to move.
        """
        buried = Parser.prose_only_calls(text)
        if buried:
            tags = "  ".join(f"<call_{name}></call_{name}>" for name in buried)
            return (
                "[System: 输出位置错误 —— 你在 <planning>（或 <reasoning>）内写了调用标签，"
                "块内的标签只是文字引用，**不会被执行**。请把调用单独放在 </planning> 之后：\n"
                f"  {tags}\n"
                "一轮只能有一个动作：一次这样的调用，或一个 <verdict> 结案。]"
            )
        return (
            f"[System: 输出格式错误 — {message} "
            f"请严格按照 SOP 格式重新输出，包含 <call_*> 或 <verdict> 标签。]"
        )

    @staticmethod
    def _state_note(evidence_chain: List[dict], served, measured_sources: Dict[str, str]) -> str:
        """
        What the session has, said plainly, after every measurement (§4.17).

        41 of the 56 acceptance sessions asked for an expert it had already
        measured (the notice was sent, the request was skipped) — so the
        result had been delivered and the model still went back for it.  The
        answer is not to hide the token but to stop making the model
        re-derive the state from a JSON blob: the executed experts, their key
        fields, and what is still available, restated after every measurement.
        The full token stays in the conversation above this line.
        """
        lines = []
        for token in evidence_chain:
            likelihood = token.get("calibrated_likelihood") or {}
            p_fake = likelihood.get("Fake")
            direction = "—"
            if p_fake is not None:
                direction = "Fake" if float(p_fake) >= 0.5 else "Real"
            lines.append(
                f"  - {token.get('source')}（{token.get('evidence_id')}）："
                f"P(Fake)={p_fake}（方向 {direction}），raw_metric={token.get('raw_metric')}，"
                f"适用 {token.get('applicability')}")
        remaining = [source for source in (served or []) if source not in measured_sources]
        return (
            "[System: 本会话状态 —— 已执行的专家：\n" + "\n".join(lines) +
            f"\n尚可调用：{', '.join(remaining) if remaining else '（无）'}。"
            "每个专家每会话只对整图测量一次：重复请求的返回值与上面已有的 token 相同，"
            "系统会跳过执行并白费一轮。请基于上面已给出的结论继续，或直接给出结案。]"
        )

    @staticmethod
    def _system_texts(conversation: List[Dict[str, str]]) -> List[str]:
        """
        The numbers the system itself put in the conversation.

        A model quoting "系统参考后验 P(Fake)=0.63" is not fabricating; only
        the notes the pipeline injected may be cited this way, never the
        expert tokens (attributing another expert's value is a defect).
        """
        return [turn.get("value", "") for turn in conversation
                if turn.get("from") == "user"
                and turn.get("value", "").startswith("[System:")]

    @staticmethod
    def _conclusion_note(decision, ledger=None,
                         turns_without_new_evidence: int = 0) -> str:
        """
        Frame the closing turn truthfully (§4.17 R2/R4).

        The old default asked the model to "explain why the existing tools
        cannot reduce the uncertainty further" — a leading question, and the
        phrasing it invited is all over the audit's `other` bucket.  The note
        now names the real reason, states what the ledger contains, and asks
        for a conclusion from those measurements or a plain statement of what
        is missing.  It never suggests a label.

        The stall wording describes what v2 actually measures: consecutive
        turns that produced no new evidence — not "the same content twice"
        (v2 never compares the texts).  The strict repeated-action semantics
        belong to v3.
        """
        summary = ledger.summary() if ledger is not None else ""
        tail = ("请只使用已执行的专家返回过的数值给出结论；若这些测量不足以判定，"
                "如实说明还缺哪种测量。")
        if decision.primary_reason == HaltingPolicyV2.BUDGET_EXHAUSTED:
            head = "[System: 取证预算已耗尽，本次会话到此结束。"
        elif HaltingPolicyV2.CONFLICT_UNRESOLVED in decision.all_reasons:
            head = "[System: 证据之间存在未解决的冲突，且不再有新的取证。"
        elif getattr(HaltingPolicyV3, "LOOP_PROTECTION", None) == decision.primary_reason:
            head = ("[System: 会话因重复或无效的动作而终止"
                    "（同一动作连续出现，或始终没有形成合法的调用/结案）。")
        elif getattr(HaltingPolicyV3, "FORMAT_FAILURE", None) == decision.primary_reason:
            head = "[System: 连续多轮没有形成合法的动作或结案，会话到此结束。"
        elif decision.primary_reason == HaltingPolicyV2.MODEL_STALLED:
            turns = getattr(decision, "stall_turns", 0) or turns_without_new_evidence
            head = (
                f"[System: 你已连续 {turns} 轮没有新增任何测量结果"
                "（判定依据是每轮是否产生新证据，不是每轮文字是否逐字相同），"
                "会话到此结束。"
            )
        else:
            head = "[System: 当前策略估计继续调用收益不足（这是策略的估计，不是对工具能力的定论）。"
        body = head if head.endswith("。") else head + "。"
        if summary:
            body += summary + "。"
        return f"{body}{tail}]"

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
        skipped_duplicate_calls: int = 0,
        policy_decision=None,
        available_experts: Optional[List[str]] = None,
        evidence_check: Optional[dict] = None,
        conclusion: Optional[dict] = None,
        v3_signals: Optional[dict] = None,
        protocol_violations: int = 0,
        condition: Optional[str] = None,
        reference_fields: bool = True,
        verdict_status: Optional[str] = None,
        verdict_label: Optional[str] = None,
    ) -> dict:
        """Assemble the observable counter block recorded in the trace."""
        counters = {
            "model_turn_count": model_turn_count,
            "expert_call_count": expert_call_count,
            "unique_evidence_count": len(evidence_chain),
            "suppressed_duplicate_count": suppressed_duplicate_count,
            # G7 entry 3: calls skipped *before* execution because the expert
            # had already measured this image.  Counted separately from the
            # post-hoc suppression so a reader can tell a wasted GPU call from
            # a call that never happened.
            "skipped_duplicate_calls": skipped_duplicate_calls,
            "weighted_cost": round(
                expert_call_count + TURN_COST_WEIGHT * model_turn_count, 4
            ),
            "policy_version": HALTING_POLICY,
            # What this session was allowed to call.  Without it a stop reason
            # like "no expected gain" cannot be audited after the fact: a
            # restricted policy and a full toolkit can stop for opposite
            # reasons and the trace looked identical (0 of 3237 traces
            # recorded this before 2026-10-11).
            "available_experts": sorted(available_experts or []),
            # Which format cell the calibration weights were read for (A2).
            "calibration_condition": condition,
            # §4.17 R9: how many turns carried more than one action and were
            # refused unexecuted.
            "protocol_violations": protocol_violations,
        }
        if policy_decision is not None:
            # A replay must be able to explain the stop from the trace alone.
            counters.update({
                "policy_reason": policy_decision.primary_reason,
                "policy_reasons": list(policy_decision.all_reasons),
                # The evidence ledger's verdict on the closing turn (§4.17 R3).
                "evidence_check_status": (evidence_check or {}).get("status"),
            })
            if reference_fields:
                counters.update({
                    "posterior": round(policy_decision.posterior, 4),
                    "conflict_score": round(policy_decision.conflict_score, 4),
                    "candidate_overridden": (
                        HaltingPolicyV2.CANDIDATE_OVERRIDDEN
                        in policy_decision.all_reasons),
                    # The estimate the reason rests on, per expert still uncalled.
                    "tool_utilities": dict(policy_decision.tool_utilities or {}),
                    # Which tokens were read at less than full weight, and after whom.
                    "redundancy_discounts": HaltingPolicyV2.discount_report(
                        evidence_chain or []),
                })
        if v3_signals:
            counters.update(v3_signals)
        if conclusion is not None:
            pass
            # §4.17 R4: one `pass` cannot stand for four different things.
            # These are the check and the format/purity of the adopted
            # conclusion; `trainable` is decided by the conversion gate.
            counters.update({
                "conclusion_source": conclusion.get("source"),
                "conclusion_verdict_parse": conclusion.get("verdict_parse"),
                "conclusion_calls_unexecuted": bool(
                    conclusion.get("calls") and not conclusion.get("calls_executed")),
                # Findings a person still has to look at (§4.17 R5)
                # A conclusion that never parsed is not an abstention: the
                # abstention rate must not absorb it (§4.17 R5).
                "verdict_status": verdict_status,
                "model_abstained": bool(
                    verdict_status == "model_concluded" and verdict_label == "Uncertain"),
                "evidence_candidates": len((evidence_check or {}).get("candidates") or []),
                "evidence_candidate_kinds": sorted({
                    finding.get("kind") for finding in
                    (evidence_check or {}).get("candidates") or []}),
            })
        return counters
