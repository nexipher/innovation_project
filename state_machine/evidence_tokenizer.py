"""
Evidence Tokenizer — converts raw expert output into structured
Evidence Token JSON ready for injection into the MLLM conversation.

Per the specification (§4.1-4.2):
  - Maps scalar strength to semantic "soft description" text.
  - Assembles the full Evidence Token Schema dict.
  - Determines support label (Real / AI-generated / Uncertain).

G1 additions (plan.md §4.8):
  - Explicit coordinate spaces: `region_pixels` + `region_normalized_1000`
    + `coordinate_space`, so pixel coordinates can never be re-read as
    normalized coordinates (and vice versa);
  - Stable `evidence_id` for deduplication and evidence-bound reporting;
  - `region_semantics` labels bboxes as diagnostic evidence regions.
"""

import hashlib
import json
from typing import List, Optional

from config import (
    NORMALIZATION_SCALE,
    REGION_SEMANTICS_DIAGNOSTIC,
    STRENGTH_THRESHOLD_LOW,
    STRENGTH_THRESHOLD_HIGH,
    STRENGTH_TEXT_MAP,
    STRENGTH_SUPPORT_MAP,
)


class EvidenceTokenizer:
    """Stateless converter: ExpertResult → Evidence Token dict."""

    @classmethod
    def evidence_id(
        cls,
        source: str,
        region_pixels: List[int],
        strength: float,
        evidence_name: str,
        measurement_scope: str = "region",
    ) -> str:
        """
        Deterministic evidence identifier.

        G3-a: a global measurement is the same measurement whatever bbox the
        model attached to it.  Keying the id on the bbox therefore let one
        whole-image measurement enter the posterior twice under two ids — and
        the halting policy weights every token, so that is a double count.
        For `measurement_scope="global"` the diagnostic region is left out of
        the key, which makes a repeat call dedup exactly like any other
        identical result; the region-scoped key is unchanged.

        Same key → same id, which makes duplicate suppression idempotent
        across runs.
        """
        parts = [str(source)]
        if measurement_scope != "global":
            parts.append(",".join(str(int(v)) for v in region_pixels))
        parts += [f"{round(float(strength), 4):.4f}", str(evidence_name)]
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
        return f"E-{digest}"

    @classmethod
    def tokenize(
        cls,
        expert_result,  # ExpertResult
        bbox: List[int],
        image_shape: tuple,
        region_normalized: Optional[List[int]] = None,
        reliability: Optional[float] = None,
        calibrated_likelihood: Optional[dict] = None,
        condition_metadata: Optional[dict] = None,
        visual_artifacts: Optional[List[str]] = None,
        semantics_aligned: Optional[bool] = None,
        applicability: Optional[str] = None,
        applicability_conditions: Optional[str] = None,
        measurement_scope: Optional[str] = None,
        region_area_ratio: Optional[float] = None,
        calibration: Optional[dict] = None,
    ) -> dict:
        """
        Build a complete Evidence Token / Evidence Bundle dict.

        Args:
            expert_result: ExpertResult dataclass from an expert's analyze().
            bbox: Absolute pixel bbox [ymin, xmin, ymax, xmax] that was analysed.
            image_shape: (height, width) of the full image.
            region_normalized: The model's requested bbox in [0, 1000] space,
                recorded for audit; None when the call did not originate from
                a normalized-space request.
            reliability: Empirical reliability (0-1) for the current condition,
                from the G2 calibration table (None when unavailable).
            calibrated_likelihood: Empirical {"Real": p, "Fake": q} for the
                measured raw metric (G2).
            condition_metadata: Format / resolution / quality condition the
                reliability was conditioned on.
            visual_artifacts: Project-relative paths of rendered artifacts.
            semantics_aligned: Whether high raw metric empirically means Fake
                (G2 calibration). False = the expert's support labels are
                inverted for this task.
            applicability: Calibration label, e.g.
                "inverted:high-metric-means-real".
            applicability_conditions: Human-readable usage conditions for this
                measurement (when it may be trusted).
            measurement_scope: "global" when the metric was measured on the
                whole image, "region" when it came from a crop.  The reliability
                table is calibrated on full images, so only global measurements
                may be read against it (G3-a).
            region_area_ratio: Diagnostic region area as a fraction of the
                image, recorded so a future crop-scale calibration can be
                conditioned on it (G3-a2).
            calibration: Provenance of the mapping that produced the numbers
                above — which group it was fitted on, on which data, with which
                weights (G7).  "The probability came from somewhere" is part of
                the evidence, not a footnote to it.

        Returns:
            Evidence Token dict matching the project schema (G2 bundle fields
            are included only when available, keeping legacy tokens lean).
        """
        region_str = f"patch_coordinates_{bbox}"
        region_pixels = [int(v) for v in bbox]

        token = {
            "evidence_id": cls.evidence_id(
                expert_result.source, region_pixels,
                expert_result.strength, expert_result.evidence_name,
                measurement_scope=measurement_scope or "region",
            ),
            "evidence_name": expert_result.evidence_name,
            "region": region_str,
            "region_pixels": region_pixels,
            "region_normalized_1000": (
                [int(v) for v in region_normalized]
                if region_normalized is not None else None
            ),
            "coordinate_space": "pixels",
            "region_semantics": REGION_SEMANTICS_DIAGNOSTIC,
            "measurement_scope": measurement_scope or "region",
            "phenomenon": expert_result.phenomenon,
            "reasoning": expert_result.reasoning,
            "strength": round(expert_result.strength, 4),
            "raw_metric": round(float(getattr(expert_result, "raw_metric", 0.0) or 0.0), 6),
            "source": expert_result.source,
            "support": expert_result.support,
            "interpretation_text": expert_result.interpretation_text,
        }

        counter_explanation = getattr(expert_result, "counter_explanation", "")
        if counter_explanation:
            token["counter_explanation"] = counter_explanation
        if reliability is not None:
            token["reliability"] = round(float(reliability), 3)
        if calibrated_likelihood:
            token["calibrated_likelihood"] = {
                key: round(float(value), 3) for key, value in calibrated_likelihood.items()
            }
        if condition_metadata or region_area_ratio is not None:
            token["condition_metadata"] = dict(condition_metadata or {})
            if region_area_ratio is not None:
                # Recorded so a future crop-scale calibration (G3-a2) can be
                # conditioned on how much of the image the region covers.
                token["condition_metadata"]["region_area_ratio"] = round(
                    float(region_area_ratio), 6
                )
        if visual_artifacts:
            token["visual_artifacts"] = list(visual_artifacts)
        # G2-c semantic transmission: direction truth and usage conditions must
        # travel with the measurement, not only the number.
        if semantics_aligned is not None:
            token["semantics_aligned"] = bool(semantics_aligned)
        if applicability:
            token["applicability"] = applicability
        if applicability_conditions:
            token["applicability_conditions"] = applicability_conditions
        if calibration:
            token["calibration"] = dict(calibration)

        return token

    @classmethod
    def strength_to_text(cls, strength: float) -> str:
        """
        Map a normalised strength value to its semantic description.

        0.0 ≤ s < 0.3 → normal hardware camera capture
        0.3 ≤ s < 0.7 → mild distortions
        0.7 ≤ s ≤ 1.0 → severe anomaly matching AI fingerprints
        """
        if strength < STRENGTH_THRESHOLD_LOW:
            return STRENGTH_TEXT_MAP["low"]
        elif strength < STRENGTH_THRESHOLD_HIGH:
            return STRENGTH_TEXT_MAP["medium"]
        else:
            return STRENGTH_TEXT_MAP["high"]

    @classmethod
    def strength_to_support(cls, strength: float) -> str:
        """Map strength → support label."""
        if strength < STRENGTH_THRESHOLD_LOW:
            return STRENGTH_SUPPORT_MAP["low"]
        elif strength < STRENGTH_THRESHOLD_HIGH:
            return STRENGTH_SUPPORT_MAP["medium"]
        else:
            return STRENGTH_SUPPORT_MAP["high"]

    # 模型可见字段：固定集合，测试锁定。多一个字段就是多一层要解释的数值。
    MODEL_VISIBLE_KEYS = ("evidence_id", "expert", "measurement", "status", "scope",
                          "score", "reading", "visual_artifacts", "artifact_meaning")

    @classmethod
    def model_visible(cls, token: dict) -> dict:
        """
        The slim pack: what the model is shown, versus what the trace keeps.

        Proposal §5's split, and the reason for it is in this project's history:
        one token used to carry `strength`, `raw_metric`, `support`, a calibrated
        likelihood, an applicability label and a direction — five numbers that
        could disagree, and did (G2-e measured frequency_v2 contradicting itself
        in 15 of 64 samples, and the model followed whichever field it trusted).

        So the model gets **one** score, with what the score-semantics registry
        says about it: its scale, its direction, the threshold that was recorded,
        and the conditions under which it may be read (NPR is strong only under
        this val's PNG encoding; UnivFD never sees the image edges).  Everything
        else stays in the token for the ledger, the converter and the reviewer.

        `status` is here because "no measurement" and "a measurement that came
        out neutral" are different facts, and a missing field must not read as
        the second one.
        """
        from utils.score_semantics import SemanticsError, direction_for_value, entry

        source = token.get("source") or ""
        if token.get("status") == "failed":
            # 失败的测量**没有分数，也没有尺度与方向**。留一个 value=None 的分数槽
            # （还带着 range/direction/threshold），模型会把它读成"一次中性的测量"，
            # 正是提案禁止的那件事——这里不给它这个槽。
            failed = {
                "evidence_id": token.get("evidence_id"),
                "expert": source,
                "measurement": token.get("evidence_name"),
                "status": "failed",
                "reading": ("本次测量执行失败，**没有任何数值**。"
                            "这不是中性结果，也不是证据——请改用别的工具或直接给出结论。"),
            }
            reason = token.get("reason")
            if reason:
                failed["reason"] = str(reason)
            return failed
        record = None
        try:
            record = entry(source)
        except SemanticsError:
            record = None

        score_field = (record or {}).get("raw_score", {}).get("field", "raw_metric")
        raw = token.get(score_field)
        if raw is None:
            raw = token.get("raw_metric", token.get("strength"))

        score: dict = {
            "name": token.get("evidence_name"),
            "value": raw,
        }
        reading_parts: list = []
        if record is None:
            # Unregistered expert (offline, mock, legacy): say so rather than
            # borrow another expert's semantics.
            score["direction"] = "未登记（无已核实的尺度与方向）"
        else:
            semantics = record["raw_score"]
            score.update({
                "range": semantics["range"],
                "direction": semantics["direction"],
                "is_calibrated_probability": False,
                "threshold": record["threshold"]["value"],
                "threshold_chosen_on": record["threshold"]["chosen_on"],
            })
            # 判定**这个值**所用的那条规则的说明排在最前：适用限制要跟着分数走，
            # 而不是躺在一段没人更新的提示词里。
            try:
                value_direction, rule = direction_for_value(source, float(raw))
                # 该值的**结论**进 score（模型据此判断这个数值能不能定方向）
                score["direction_for_this_value"] = value_direction
                if rule.get("prompt_note"):
                    reading_parts.append(rule["prompt_note"])
            except (SemanticsError, TypeError, ValueError):
                pass
            # 只有给模型看的那一句（prompt_note）进包。`usage_conditions` 与
            # `limits` 是**内部**说明（含 bootstrap 数值、写法要求），把它们拼进
            # 模型输入就是把我们写给自己的话当成证据条件——这一处曾经就是这样。
            if record.get("prompt_note"):
                reading_parts.append(record["prompt_note"])

        visible = {
            "evidence_id": token.get("evidence_id"),
            "expert": source,
            "measurement": token.get("evidence_name"),
            "status": token.get("status", "measured"),
            "scope": token.get("measurement_scope"),
            "score": score,
            "reading": (
                (token.get("phenomenon") or "").strip()
                + ((" " + " ".join(reading_parts)) if reading_parts else "")
            ).strip(),
        }
        artifacts = token.get("visual_artifacts")
        if artifacts:
            visible["visual_artifacts"] = list(artifacts)
            visible["artifact_meaning"] = (
                "诊断用可视化产物（供人核对，不是模型的判据；数值以上面的 score 为准）")
        return visible

    @classmethod
    def to_json(cls, token: dict) -> str:
        """Serialize an Evidence Token dict to a JSON string for MLLM injection."""
        return json.dumps(token, ensure_ascii=False)
