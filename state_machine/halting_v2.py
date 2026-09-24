"""
Halting policy v2 — decide from the posterior, not from tag order (G3-c).

v1 halted on the first `<verdict>` tag, compared `strength` across experts that
do not share a scale, and read information gain off the difference between two
adjacent strengths.  After G2-e a high strength even means opposite things for
different experts, so those comparisons are not merely noisy — they are wrong.

v2 keeps a probability and asks what another call would buy:

    posterior         reliability-weighted log-odds over rectified tokens
    conflict_score    how much the weighted contributions disagree
    tool utilities    per-expert expected gain, discounted by redundancy
    call cost         tokens, turns and calls already spent

Decision rules, in order:

  1. no evidence yet, budget available         -> continue (explore)
  2. the model's <verdict> is a *candidate*    -> accepted only when the
     posterior agrees, it is confident enough and no strong conflict is open;
     otherwise it is recorded as overridden
  3. the model repeats an overridden candidate -> halt (model_stalled): the
     policy can recommend a tool but not compel it, and a loop that keeps
     asking will only burn turns until the budget does the same job
  4. budget exhausted                          -> halt; a label is taken from
     the posterior only if it is confident, never from the exhaustion itself
  5. no tool has positive expected net utility -> halt (no_expected_gain)
  6. otherwise                                 -> continue, recommending the
     expert with the highest expected net utility

Every return path fills `all_reasons`, so a replay can explain any decision
from the trace alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from config import (
    MAX_EXPERT_CALLS,
    MAX_MODEL_TURNS,
    POLICY_CALL_COST,
    POLICY_CONFLICT_TOLERANCE,
    POLICY_CONFIDENT_POSTERIOR,
    POLICY_STEP_LOG_ODDS,
)


@dataclass
class HaltingDecision:
    """Why the loop continues or stops, and with what conclusion."""

    action: str                      # "continue" | "halt"
    verdict: Optional[str] = None    # Real / Fake / Uncertain when halting
    confidence: Optional[float] = None
    primary_reason: str = ""
    all_reasons: List[str] = field(default_factory=list)
    next_expert: Optional[str] = None
    posterior: float = 0.5           # P(Fake)
    conflict_score: float = 0.0
    expected_tool_gain: float = 0.0
    expected_net_utility: float = 0.0


class HaltingPolicyV2:
    """Posterior-driven halting policy."""

    # Reasons (stable strings, recorded in the trace)
    NO_EVIDENCE = "no_evidence_yet"
    CANDIDATE_ACCEPTED = "candidate_matches_posterior"
    CANDIDATE_OVERRIDDEN = "candidate_overridden_by_posterior"
    CONFLICT_UNRESOLVED = "conflict_unresolved"
    MODEL_STALLED = "model_stalled"
    CANDIDATE_WITHOUT_TOOLS = "candidate_without_tools"
    AWAITING_VERDICT = "awaiting_verdict"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_EXPECTED_GAIN = "no_expected_gain"
    EXPLORING = "exploring"

    # ------------------------------------------------------------------
    # Posterior and conflict
    # ------------------------------------------------------------------

    @staticmethod
    def token_weight(token: dict) -> float:
        """
        How much this token may move the posterior.

        The calibration table's `reliability` is the polarity-corrected
        separation (an AUROC), so the Youden-style margin `2*AUROC - 1` is the
        share of its ranking power that is better than chance.  Uncalibrated
        tokens carry no measured weight and therefore do not move the
        posterior: an expert's own claim is not evidence about its accuracy.
        """
        reliability = token.get("reliability")
        if reliability is None:
            likelihood = token.get("calibrated_likelihood") or {}
            if not likelihood:
                return 0.0
            return 0.0
        return max(0.0, 2.0 * float(reliability) - 1.0)

    @classmethod
    def token_contribution(cls, token: dict) -> float:
        """Signed log-odds this token contributes (+ towards Fake)."""
        likelihood = (token.get("calibrated_likelihood") or {}).get("Fake")
        if likelihood is None:
            return 0.0
        weight = cls.token_weight(token)
        if weight == 0.0:
            return 0.0
        signed = 2.0 * float(likelihood) - 1.0
        return weight * signed * POLICY_STEP_LOG_ODDS

    @classmethod
    def posterior(cls, evidence_chain: List[dict]) -> float:
        """P(Fake) from the accumulated weighted evidence (prior 0.5)."""
        log_odds = sum(cls.token_contribution(t) for t in evidence_chain)
        return float(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, log_odds)))))

    @classmethod
    def conflict_score(cls, evidence_chain: List[dict]) -> float:
        """
        Weighted disagreement in [0, 1]: 0 when every token points the same
        way, 1 when the signed contributions cancel exactly.
        """
        contributions = [cls.token_contribution(t) for t in evidence_chain]
        total_magnitude = sum(abs(c) for c in contributions)
        if total_magnitude == 0.0:
            return 0.0
        return float(1.0 - abs(sum(contributions)) / total_magnitude)

    # ------------------------------------------------------------------
    # Tool utility
    # ------------------------------------------------------------------

    @staticmethod
    def expected_gain(expert_weight: float, posterior: float) -> float:
        """
        What one more call by this expert is expected to buy.

        A tool is worth most when the question is still open (posterior near
        0.5) and it has measured discriminative power to spend.
        """
        uncertainty = 1.0 - abs(2.0 * posterior - 1.0)
        return float(expert_weight * 0.5 * uncertainty)

    @classmethod
    def tool_utilities(
        cls,
        evidence_chain: List[dict],
        available_experts: Dict[str, float],
        called_experts: set,
    ) -> Dict[str, float]:
        """
        Expected net utility per expert: gain minus call cost.

        Repeating an expert cannot help — the same image yields the same
        measurement, and identical tokens are suppressed upstream — so already
        called experts are priced at zero gain.
        """
        posterior = cls.posterior(evidence_chain)
        utilities = {}
        for name, weight in available_experts.items():
            if name in called_experts:
                utilities[name] = -POLICY_CALL_COST
                continue
            utilities[name] = cls.expected_gain(weight, posterior) - POLICY_CALL_COST
        return utilities

    # ------------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------------

    @classmethod
    def decide(
        cls,
        model_turns: int,
        expert_calls: int,
        evidence_chain: List[dict],
        candidate_verdict: Optional[dict],
        available_experts: Dict[str, float],
        called_experts: set,
        turns_without_new_evidence: int = 0,
        tools_available: bool = True,
    ) -> HaltingDecision:
        """
        Evaluate the current state and return the next action.

        Args:
            model_turns / expert_calls: current counters.
            evidence_chain: unique, rectified evidence tokens.
            candidate_verdict: the model's <verdict> this turn, if any.
            available_experts: expert name -> measured weight (2*AUROC-1).
            called_experts: experts already used in this session.
            turns_without_new_evidence: model turns since the last new token;
                two in a row with an overridden candidate means the model is
                not going to take the recommendation.
            tools_available: False for a session whose protocol forbids tool
                calls (the no-tool RGB baseline).  Nothing will ever reduce
                the posterior there, so the policy must judge the session on
                the model's own verdict instead of asking for evidence that
                cannot arrive.
        """
        posterior = cls.posterior(evidence_chain)
        conflict = cls.conflict_score(evidence_chain)
        utilities = cls.tool_utilities(evidence_chain, available_experts, called_experts)
        best_expert = max(utilities, key=utilities.get) if utilities else None
        best_utility = utilities.get(best_expert, 0.0) if best_expert else 0.0

        decision = HaltingDecision(
            action="continue",
            posterior=posterior,
            conflict_score=conflict,
            expected_tool_gain=best_utility + (POLICY_CALL_COST if best_expert else 0.0),
            expected_net_utility=best_utility,
            next_expert=best_expert,
        )

        confident = abs(2.0 * posterior - 1.0) >= abs(2.0 * POLICY_CONFIDENT_POSTERIOR - 1.0)
        label = "Fake" if posterior >= 0.5 else "Real"
        conflict_open = conflict > POLICY_CONFLICT_TOLERANCE

        # 0. a protocol that forbids tool calls: there is no evidence to wait
        #    for, so the model's own conclusion is the measurement
        if not tools_available:
            # A tool recommendation would be noise here, not advice.
            decision.next_expert = None
            decision.expected_tool_gain = 0.0
            decision.expected_net_utility = 0.0
            if candidate_verdict and candidate_verdict.get("verdict") in (
                "Real", "Fake", "Uncertain",
            ):
                decision.action = "halt"
                decision.verdict = candidate_verdict["verdict"]
                decision.confidence = round(
                    float(candidate_verdict.get("confidence") or 0.5), 4
                )
                decision.primary_reason = cls.CANDIDATE_WITHOUT_TOOLS
                decision.all_reasons = [cls.CANDIDATE_WITHOUT_TOOLS]
                return decision
            if cls._budget_exhausted(model_turns, expert_calls):
                return cls._halt(decision, "Uncertain", 0.5, cls.BUDGET_EXHAUSTED,
                                 [cls.BUDGET_EXHAUSTED])
            decision.primary_reason = cls.AWAITING_VERDICT
            decision.all_reasons = [cls.AWAITING_VERDICT]
            return decision

        # 1. nothing measured yet — with budget left, measure something
        if not evidence_chain and not cls._budget_exhausted(model_turns, expert_calls):
            decision.primary_reason = cls.EXPLORING
            decision.all_reasons = [cls.NO_EVIDENCE, cls.EXPLORING]
            return decision

        # 2. the model proposed a conclusion: it is a candidate, not a decision
        if candidate_verdict:
            proposed = candidate_verdict.get("verdict")
            agrees = proposed in (None, "Uncertain") or proposed == label
            if agrees and confident and not conflict_open:
                return cls._halt(decision, label, posterior,
                                 cls.CANDIDATE_ACCEPTED, [cls.CANDIDATE_ACCEPTED])
            decision.all_reasons.append(cls.CANDIDATE_OVERRIDDEN)

        # 3. the model keeps proposing the same rejected conclusion: the
        #    policy can recommend but not compel, so waiting is just a slower
        #    way to reach the budget
        if candidate_verdict and turns_without_new_evidence >= 2:
            reasons = decision.all_reasons + [cls.MODEL_STALLED]
            if conflict_open:
                reasons.append(cls.CONFLICT_UNRESOLVED)
                return cls._halt(decision, "Uncertain", 0.5, cls.MODEL_STALLED, reasons)
            if confident:
                return cls._halt(decision, label, posterior, cls.MODEL_STALLED, reasons)
            return cls._halt(decision, "Uncertain", 0.5, cls.MODEL_STALLED, reasons)

        # 4. no resources left: the exhaustion decides nothing by itself, and
        #    an unresolved conflict is recorded rather than hidden behind it
        if cls._budget_exhausted(model_turns, expert_calls):
            reasons = decision.all_reasons + [cls.BUDGET_EXHAUSTED]
            if confident and not conflict_open:
                return cls._halt(decision, label, posterior, cls.BUDGET_EXHAUSTED, reasons)
            if conflict_open:
                reasons = reasons + [cls.CONFLICT_UNRESOLVED]
            return cls._halt(decision, "Uncertain", 0.5, cls.BUDGET_EXHAUSTED, reasons)

        # 5. further evidence cannot pay for itself
        if best_utility <= 0.0:
            reasons = decision.all_reasons + [cls.NO_EXPECTED_GAIN]
            if conflict_open:
                reasons.append(cls.CONFLICT_UNRESOLVED)
                return cls._halt(decision, "Uncertain", 0.5, cls.NO_EXPECTED_GAIN, reasons)
            if confident:
                return cls._halt(decision, label, posterior, cls.NO_EXPECTED_GAIN, reasons)
            return cls._halt(decision, "Uncertain", 0.5, cls.NO_EXPECTED_GAIN, reasons)

        # 6. keep going; say which tool is worth the next call
        if not decision.all_reasons:
            decision.all_reasons = [cls.EXPLORING]
        decision.primary_reason = decision.all_reasons[0]
        return decision

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _budget_exhausted(model_turns: int, expert_calls: int) -> bool:
        return expert_calls >= MAX_EXPERT_CALLS or model_turns >= MAX_MODEL_TURNS

    @staticmethod
    def _halt(decision: HaltingDecision, verdict: str, confidence: float,
              reason: str, reasons: List[str]) -> HaltingDecision:
        decision.action = "halt"
        decision.verdict = verdict
        decision.confidence = round(float(confidence), 4)
        decision.primary_reason = reason
        decision.all_reasons = reasons
        return decision

    @staticmethod
    def expert_weights(reliability_entries: Dict[str, Optional[float]]) -> Dict[str, float]:
        """
        Turn the calibration table's separations into policy weights.

        `reliability` is the polarity-corrected AUROC under the format-matched
        conditions, so the Youden margin is what the expert can still tell
        apart after every confound G2-b could remove.
        """
        return {
            name: max(0.0, 2.0 * float(sep) - 1.0)
            for name, sep in reliability_entries.items()
            if sep is not None
        }
