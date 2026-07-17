"""
Halting criteria checker for the forensic state machine.

Implements four independent halting guards, checked in priority order:
  1. Verdict output  — MLLM produced a <verdict> tag.
  2. Max steps       — hard cap of MAX_STEPS expert-call iterations.
  3. Evidence conflict — contradictory expert opinions detected.
  4. Information gain — KL divergence between successive confidence
     distributions fell below threshold.

All methods are stateless; the check() method takes the current session
state as arguments.
"""

import numpy as np
from typing import List, Optional, Tuple

from config import MAX_STEPS, ENTROPY_THRESHOLD, KL_THRESHOLD


class HaltingChecker:
    """Stateless halting-criteria evaluator."""

    # Valid halting reasons
    VERDICT_OUTPUT = "verdict_output"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    EVIDENCE_CONFLICT = "evidence_conflict"
    INFO_GAIN_CONVERGED = "info_gain_converged"

    @classmethod
    def check(
        cls,
        step: int,
        evidence_chain: List[dict],
        last_output: str,
    ) -> Tuple[bool, str]:
        """
        Evaluate all halting criteria in priority order.
        Returns (should_halt: bool, reason: str).
        """
        # 1. Did the model output a verdict?
        verdict = cls._verdict_detected(last_output)
        if verdict:
            return True, cls.VERDICT_OUTPUT

        # 2. Max steps reached?
        if cls._max_steps_reached(step):
            return True, cls.MAX_STEPS_EXCEEDED

        # 3. Evidence conflict?
        if cls._conflict_detected(evidence_chain):
            return True, cls.EVIDENCE_CONFLICT

        # 4. Information gain converged?
        if cls._info_gain_converged(evidence_chain):
            return True, cls.INFO_GAIN_CONVERGED

        return False, ""

    # ------------------------------------------------------------------
    # Individual criteria
    # ------------------------------------------------------------------

    @classmethod
    def _verdict_detected(cls, output: str) -> Optional[str]:
        """Check if the output contains a <verdict> tag with valid JSON."""
        from utils.parser import Parser
        verdict = Parser.parse_verdict(output)
        if verdict and "verdict" in verdict:
            return verdict["verdict"]
        return None

    @classmethod
    def _max_steps_reached(cls, step: int) -> bool:
        """Hard cap: step count >= MAX_STEPS."""
        return step >= MAX_STEPS

    @classmethod
    def _conflict_detected(cls, evidence_chain: List[dict]) -> bool:
        """
        Detect strong contradictory evidence.

        Condition: at least two experts disagree fundamentally —
        one strongly says AI-generated (strength > 0.7),
        another strongly says Real (strength < 0.3).
        """
        if len(evidence_chain) < 2:
            return False

        has_strong_fake = False
        has_strong_real = False

        for ev in evidence_chain:
            strength = ev.get("strength", 0.5)
            if strength > 0.7:
                has_strong_fake = True
            if strength < 0.3:
                has_strong_real = True

        return has_strong_fake and has_strong_real

    @classmethod
    def _info_gain_converged(cls, evidence_chain: List[dict]) -> bool:
        """
        Check whether new evidence has stopped changing the model's confidence.

        Uses the change in strength values between the last two pieces of
        evidence as a proxy for information gain.  If the difference is below
        KL_THRESHOLD, the loop has converged.

        Requires at least 2 evidence entries to compare.
        """
        if len(evidence_chain) < 2:
            return False

        # Compare the last two evidence strengths
        s_prev = evidence_chain[-2].get("strength", 0.5)
        s_curr = evidence_chain[-1].get("strength", 0.5)

        delta = abs(s_curr - s_prev)
        return delta < KL_THRESHOLD * 100  # scale up for direct strength comparison

    # ------------------------------------------------------------------
    # Utility: compute classification entropy
    # ------------------------------------------------------------------

    @classmethod
    def compute_entropy(cls, probs: List[float]) -> float:
        """
        Compute Shannon entropy of a probability distribution.

        Args:
            probs: List of probabilities that sum to ~1.0.

        Returns:
            Entropy in nats (using natural log).
        """
        probs = np.array(probs, dtype=np.float64)
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum()  # re-normalise
        return float(-np.sum(probs * np.log(probs)))

    @classmethod
    def compute_kl_divergence(cls, p: List[float], q: List[float]) -> float:
        """
        Compute KL divergence D_KL(P || Q).

        Args:
            p, q: Probability distributions over the same categories.
        """
        p = np.array(p, dtype=np.float64)
        q = np.array(q, dtype=np.float64)
        p = np.clip(p, 1e-12, 1.0)
        q = np.clip(q, 1e-12, 1.0)
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))
