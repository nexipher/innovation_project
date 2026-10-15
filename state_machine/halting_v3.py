"""
Halting policy v3 — the model decides, the reference advises (G7 §4.17 R5).

v2 owned the label: a candidate the posterior disagreed with was overridden
until the model repeated itself, and an `Uncertain` candidate counted as
agreement with a confident posterior, which then put *the posterior's* label
on the session.  The A-segment audit found what that produced: 68 sessions
whose model conclusion was silently replaced, 7 the other way round, one
label swapped for the other.

v3 does not decide the label at all.  Its job is to say **when the session
ends and why**:

    model_concluded     the model produced a legal conclusion — that is the
                        verdict, after the evidence ledger has checked it
    budget_exhausted    the call/turn budget ran out; the closing turn still
                        has to produce the conclusion
    loop_protection     the same action repeated, or turns with no legal
                        action at all — v2 called this "model_stalled" while
                        measuring something else ("no new evidence")
    format_failure      turns that never became a legal action or verdict
    evidence_check_failed  recorded by the controller, not decided here

Stage 4.2 (2026-10-15): the reference posterior, the conflict score and the tool
utilities are **not computed here any more**.  They used to be written beside the
model's label as a reference; the decision was already the model's, but keeping
the numbers alive kept the calibration table loaded and kept a v2-shaped object
in every v3 trace — and a number that exists tends to get read.  Offline
comparison can compute them deliberately; a session does not.

`FORENSIC_HALTING_POLICY=v2` keeps that machinery (and its halting rules) exactly
as it was, for replaying the batches it produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import (
    MAX_EXPERT_CALLS,
    MAX_MODEL_TURNS,
    POLICY_CONFLICT_TOLERANCE,
)
from state_machine.halting_v2 import HaltingDecision, HaltingPolicyV2

# `confidence` in a v3 verdict is the model's own statement about its
# conclusion.  It is not a calibrated probability, and nothing downstream may
# read it as one — the calibrated numbers live in the evidence tokens.
MODEL_CONFIDENCE_FALLBACK = 0.5


@dataclass
class V3Signals:
    """What the controller observed about this turn, for the loop guards."""

    repeated_actions: int = 0     # consecutive turns with the same action
    invalid_actions: int = 0      # consecutive turns with no legal action
    format_failures: int = 0      # malformed turns in the session so far


class HaltingPolicyV3:
    """Stop reasons are the policy's output; the label is the model's."""

    # Stop reasons (stable strings, recorded in the trace)
    MODEL_CONCLUDED = "model_concluded"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LOOP_PROTECTION = "loop_protection"
    FORMAT_FAILURE = "format_failure"
    EVIDENCE_CHECK_FAILED = "evidence_check_failed"

    # Continue reasons
    EXPLORING = "exploring"
    AWAITING_ACTION = "awaiting_action"
    ILLEGAL_CONCLUSION = "illegal_conclusion"

    # The same action three turns running is a loop; three turns that never
    # became an action are a format failure, not exploration.
    REPEATED_ACTION_LIMIT = 2
    INVALID_ACTION_LIMIT = 3

    LEGAL_LABELS = ("Real", "Fake", "Uncertain")

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    @classmethod
    def decide(
        cls,
        model_turns: int,
        expert_calls: int,
        evidence_chain: List[dict],
        candidate_verdict: Optional[dict] = None,
        available_experts: Optional[Dict[str, float]] = None,
        called_experts: Optional[set] = None,
        repeated_actions: int = 0,
        invalid_actions: int = 0,
        tools_available: bool = True,
        compute_reference: bool = False,
    ) -> HaltingDecision:
        """
        Decide whether the session ends, and record the reference beside it.

        A legal conclusion ends the session immediately: the evidence check in
        the controller still has to accept the *text*, but the label is not
        this policy's to change.
        """
        # 参考后验、冲突度、预计收益只在 v1/v2 复算里计算（提案 §7，阶段 4.2）：
        # v3 一次都不算——不算就没有可注入的东西，也就不会有人误读它。
        if compute_reference:
            posterior = HaltingPolicyV2.posterior(evidence_chain)
            conflict = HaltingPolicyV2.conflict_score(evidence_chain)
            utilities = HaltingPolicyV2.tool_utilities(
                evidence_chain, available_experts, called_experts)
            best_expert = max(utilities, key=utilities.get) if utilities else None
            best_utility = utilities.get(best_expert, 0.0) if best_expert else 0.0
            decision = HaltingDecision(
                action="continue",
                posterior=posterior,
                tool_utilities=utilities,
                conflict_score=conflict,
                expected_tool_gain=best_utility + (
                    HaltingPolicyV2.call_cost(best_expert) if best_expert else 0.0),
                expected_net_utility=best_utility,
                next_expert=best_expert if tools_available else None,
                reference_label=(("Fake" if posterior >= 0.5 else "Real")
                                 if evidence_chain else None),
            )
        else:
            decision = HaltingDecision(action="continue")

        proposed = (candidate_verdict or {}).get("verdict")
        if proposed in cls.LEGAL_LABELS:
            decision.action = "halt"
            decision.verdict = proposed
            decision.confidence = cls._confidence(candidate_verdict.get("confidence"))
            decision.primary_reason = cls.MODEL_CONCLUDED
            decision.all_reasons = [cls.MODEL_CONCLUDED]
            if compute_reference:
                decision.advisory = cls.advisory_notes(
                    decision.reference_label, decision.conflict_score,
                    decision.expected_net_utility, proposed, tools_available)
            return decision

        if candidate_verdict:
            # A <verdict> tag whose label is not Real / Fake / Uncertain: a
            # protocol problem, answered as one.
            decision.primary_reason = cls.ILLEGAL_CONCLUSION
            decision.all_reasons = [cls.ILLEGAL_CONCLUSION]
            return decision

        if invalid_actions >= cls.INVALID_ACTION_LIMIT:
            decision.all_reasons = [cls.FORMAT_FAILURE,
                                    f"invalid_actions={invalid_actions}"]
            return cls._stop(decision, cls.FORMAT_FAILURE)
        if repeated_actions >= cls.REPEATED_ACTION_LIMIT:
            decision.all_reasons = [cls.LOOP_PROTECTION,
                                    f"repeated_actions={repeated_actions}"]
            return cls._stop(decision, cls.LOOP_PROTECTION)
        if model_turns >= MAX_MODEL_TURNS or expert_calls >= MAX_EXPERT_CALLS:
            decision.all_reasons = [cls.BUDGET_EXHAUSTED]
            return cls._stop(decision, cls.BUDGET_EXHAUSTED)

        if not evidence_chain:
            decision.primary_reason = cls.EXPLORING
            decision.all_reasons = [cls.EXPLORING]
            return decision

        # Nothing to stop for: keep going, advise, never compel.
        decision.primary_reason = cls.AWAITING_ACTION
        decision.all_reasons = [cls.AWAITING_ACTION]
        if compute_reference:
            decision.advisory = cls.advisory_notes(
                decision.reference_label, decision.conflict_score,
                decision.expected_net_utility, None, tools_available)
        return decision

    # ------------------------------------------------------------------
    # Reference, kept separate from the decision
    # ------------------------------------------------------------------

    @classmethod
    def advisory_notes(cls, reference_label: Optional[str], conflict: float,
                       best_utility: float, proposed: Optional[str],
                       tools_available: bool) -> List[str]:
        """
        What the reference would say — recorded, never imposed.

        `reference_no_expected_gain` is the one v2 turned into a stop rule and
        a refused call; in v3 it is a note in the trace.
        """
        notes: List[str] = []
        if tools_available and best_utility <= 0.0:
            notes.append("reference_no_expected_gain")
        if conflict > POLICY_CONFLICT_TOLERANCE:
            notes.append("reference_conflict_unresolved")
        if proposed in ("Real", "Fake"):
            if reference_label is not None and reference_label != proposed:
                notes.append("reference_disagrees")
        elif proposed == "Uncertain":
            # Not agreement: the model abstained, and the record says so.
            notes.append("model_abstained")
        return notes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _confidence(value) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return MODEL_CONFIDENCE_FALLBACK
        return round(min(max(number, 0.0), 1.0), 4)

    @staticmethod
    def _stop(decision: HaltingDecision, reason: str) -> HaltingDecision:
        """
        Halt to *ask for a conclusion*, not to supply one.

        No verdict is written here: the closing turn has to produce it, and if
        that turn does not parse the session is recorded as unparseable rather
        than given a label of the policy's choosing.
        """
        decision.action = "halt"
        decision.verdict = None
        decision.confidence = None
        decision.primary_reason = reason
        return decision
