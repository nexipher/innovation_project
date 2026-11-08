import pytest
"""Unit tests for Evidence Tokenizer."""

from state_machine.evidence_tokenizer import EvidenceTokenizer

pytestmark = pytest.mark.core

class TestStrengthMapping:
    def test_low(self):
        text = EvidenceTokenizer.strength_to_text(0.0)
        assert "normal" in text.lower() or "hardware" in text.lower()
        assert EvidenceTokenizer.strength_to_support(0.0) == "Real"

    def test_medium(self):
        text = EvidenceTokenizer.strength_to_text(0.5)
        assert "mild" in text.lower() or "distortion" in text.lower()
        assert EvidenceTokenizer.strength_to_support(0.5) == "Uncertain"

    def test_high(self):
        text = EvidenceTokenizer.strength_to_text(0.9)
        assert "severe" in text.lower() or "anomaly" in text.lower()
        assert EvidenceTokenizer.strength_to_support(0.9) == "AI-generated"

    def test_boundaries(self):
        """Test mapping at exact boundary values."""
        assert EvidenceTokenizer.strength_to_support(0.3) == "Uncertain"
        assert EvidenceTokenizer.strength_to_support(0.7) == "AI-generated"

class TestTokenize:
    def test_tokenize_from_result(self):
        """Build a token from a minimal ExpertResult-like object."""
        class FakeResult:
            evidence_name = "test_evidence"
            phenomenon = "test phenomenon"
            reasoning = "test reasoning"
            strength = 0.85
            source = "test_expert"
            support = "AI-generated"
            interpretation_text = "severe anomaly"
        token = EvidenceTokenizer.tokenize(FakeResult(), [10, 20, 30, 40], (100, 200))
        assert token["evidence_name"] == "test_evidence"
        assert token["region"] == "patch_coordinates_[10, 20, 30, 40]"
        assert token["strength"] == 0.85
        assert token["source"] == "test_expert"

    def test_to_json(self):
        token = {"evidence_name": "test", "strength": 0.5}
        json_str = EvidenceTokenizer.to_json(token)
        assert '"evidence_name"' in json_str
        assert '"test"' in json_str

class TestTruForVisiblePack:
    """TruFor answers the local question, not the whole-image question."""

    @staticmethod
    def _token(score):
        return {
            "evidence_id": "E-test",
            "evidence_name": "manipulation_anomaly_score",
            "source": "trufor_expert",
            "status": "measured",
            "measurement_scope": "global",
            "raw_metric": score,
            "phenomenon": "local score",
            "image_shape": [333, 500],
            "visual_artifacts": ["anomaly.png", "conf.png"],
        }

    def test_score_direction_is_value_specific_and_local_only(self):
        high = EvidenceTokenizer.model_visible(self._token(0.72))["score"]
        low = EvidenceTokenizer.model_visible(self._token(0.23))["score"]
        assert high["direction"] == "higher_is_local_manipulation"
        assert "above" in high["direction_for_this_value"]
        assert "below" in low["direction_for_this_value"]
        for score in (high, low):
            assert "NOT a whole-image AI-generation judgment" in \
                score["direction_for_this_value"]

    def test_map_coordinates_are_named_and_contain_no_pixel_size(self):
        visible = EvidenceTokenizer.model_visible(self._token(0.72))
        assert visible["map_coordinates"]["space"] == "original_image"
        assert visible["map_coordinates"]["bbox_format"] == \
            "[ymin, xmin, ymax, xmax]"
        assert visible["map_coordinates"]["bbox_units"] == "normalized_0_1000"
        assert "image_size" not in visible["map_coordinates"]
        assert "not bbox" in visible["map_coordinates"]["note"]


class TestEvidenceId:
    """G1 (§4.8): deterministic evidence ids for deduplication."""

    def test_same_inputs_same_id(self):
        a = EvidenceTokenizer.evidence_id("noise_expert", [10, 20, 30, 40], 0.7631, "noise_residual_inconsistency")
        b = EvidenceTokenizer.evidence_id("noise_expert", [10, 20, 30, 40], 0.7631, "noise_residual_inconsistency")
        assert a == b

    def test_different_region_different_id(self):
        a = EvidenceTokenizer.evidence_id("noise_expert", [10, 20, 30, 40], 0.5, "x")
        b = EvidenceTokenizer.evidence_id("noise_expert", [11, 20, 30, 40], 0.5, "x")
        assert a != b

    def test_different_strength_different_id(self):
        a = EvidenceTokenizer.evidence_id("noise_expert", [10, 20, 30, 40], 0.5, "x")
        b = EvidenceTokenizer.evidence_id("noise_expert", [10, 20, 30, 40], 0.5002, "x")
        assert a != b

    def test_id_format(self):
        eid = EvidenceTokenizer.evidence_id("jpeg_expert", [0, 0, 1, 1], 0.1, "y")
        assert eid.startswith("E-")
        assert len(eid) == 12

    def test_global_measurement_ignores_the_diagnostic_region(self):
        """
        G3-a: the expert measures the whole image, so the model's bbox is a
        diagnostic annotation, not part of the measurement identity.  If it
        keyed the id, one measurement could enter the posterior twice — and
        the halting policy weights every token.
        """
        a = EvidenceTokenizer.evidence_id(
            "noise_expert", [10, 20, 30, 40], 0.5, "x", measurement_scope="global")
        b = EvidenceTokenizer.evidence_id(
            "noise_expert", [400, 300, 700, 600], 0.5, "x", measurement_scope="global")
        assert a == b

    def test_region_measurement_still_keys_on_the_region(self):
        a = EvidenceTokenizer.evidence_id(
            "noise_expert", [10, 20, 30, 40], 0.5, "x", measurement_scope="region")
        b = EvidenceTokenizer.evidence_id(
            "noise_expert", [400, 300, 700, 600], 0.5, "x", measurement_scope="region")
        assert a != b

    def test_scope_is_part_of_the_identity(self):
        """A crop measurement and a whole-image one are different evidence."""
        global_id = EvidenceTokenizer.evidence_id(
            "noise_expert", [10, 20, 30, 40], 0.5, "x", measurement_scope="global")
        region_id = EvidenceTokenizer.evidence_id(
            "noise_expert", [10, 20, 30, 40], 0.5, "x", measurement_scope="region")
        assert global_id != region_id

class TestTokenizeG1Fields:
    """G1 (§4.8): dual-space coordinates and region semantics in the token."""

    class FakeResult:
        evidence_name = "noise_residual_inconsistency"
        phenomenon = "phenomenon"
        reasoning = "reasoning"
        strength = 0.7631
        source = "noise_expert"
        support = "AI-generated"
        interpretation_text = "severe anomaly"

    def test_token_carries_both_coordinate_spaces(self):
        token = EvidenceTokenizer.tokenize(
            self.FakeResult(), [150, 75, 600, 375], (750, 500),
            region_normalized=[200, 150, 800, 750],
        )
        assert token["region_pixels"] == [150, 75, 600, 375]
        assert token["region_normalized_1000"] == [200, 150, 800, 750]
        assert token["coordinate_space"] == "pixels"
        assert token["region_semantics"] == "diagnostic_evidence_region"

    def test_token_without_normalized_request(self):
        token = EvidenceTokenizer.tokenize(self.FakeResult(), [0, 0, 10, 10], (100, 100))
        assert token["region_normalized_1000"] is None

    def test_token_has_stable_evidence_id(self):
        token = EvidenceTokenizer.tokenize(self.FakeResult(), [10, 20, 30, 40], (100, 200))
        expected = EvidenceTokenizer.evidence_id(
            "noise_expert", [10, 20, 30, 40], 0.7631, "noise_residual_inconsistency")
        assert token["evidence_id"] == expected

    def test_legacy_region_string_kept(self):
        token = EvidenceTokenizer.tokenize(self.FakeResult(), [10, 20, 30, 40], (100, 200))
        assert token["region"] == "patch_coordinates_[10, 20, 30, 40]"


def test_trufor_visible_pack_carries_raw_band_and_artifact_layout():
    token = {
        "evidence_id": "E-band",
        "evidence_name": "manipulation_anomaly_score",
        "source": "trufor_expert",
        "status": "measured",
        "measurement_scope": "global",
        "raw_metric": 0.72,
        "phenomenon": "局部篡改检测分数",
        "image_shape": [64, 80],
        "condition_metadata": {},
        "visual_artifacts": ["anomaly.png"],
        "artifact_layout": {"anomaly_map": {"content_box": [0, 38, 80, 102]}},
    }
    pack = EvidenceTokenizer.model_visible(token)
    assert pack["score"]["raw_band"]["label"] == "较高"
    assert pack["artifact_layout"]["anomaly_map"]["content_area"] == \
        "full original-image frame"
    assert "content_box" not in pack["artifact_layout"]["anomaly_map"]
    assert "content_box_normalized_1000" not in pack["artifact_layout"]["anomaly_map"]


def test_trufor_token_carries_map_stats_from_nested_stats():
    class FakeResult:
        evidence_name = "manipulation_anomaly_score"
        phenomenon = "局部篡改检测分数"
        reasoning = "reasoning"
        strength = 0.7
        raw_metric = 0.7
        source = "trufor_expert"
        support = "Uncertain"
        interpretation_text = "text"
        metadata = {
            "stats": {"anomaly_peak_normalized_1000": [10, 20],
                      "high_response_fraction": 0.1},
            "map_size": [32, 48],
        }
    token = EvidenceTokenizer.tokenize(FakeResult(), [0, 0, 10, 10], (64, 80))
    assert token["map_stats"]["anomaly_peak_normalized_1000"] == [10, 20]
    assert token["map_size"] == [32, 48]


def test_visible_layout_does_not_offer_full_frame_as_a_bbox():
    token = {
        "evidence_id": "E-full", "evidence_name": "manipulation_anomaly_score",
        "source": "trufor_expert", "status": "measured",
        "measurement_scope": "global", "raw_metric": 0.7,
        "phenomenon": "局部篡改检测分数", "image_shape": [64, 80],
        "condition_metadata": {"stats": {}},
        "visual_artifacts": ["traces/evidence/x/model_anomaly_map.png"],
        "model_image_layout": {"anomaly_map": {
            "title": "TruFor anomaly map — full frame",
            "content_box": [0, 38, 80, 102], "canvas_size": [285, 164],
            "low_colour": "blue", "high_colour": "red",
            "low_label": "LOW response (blue)",
            "high_label": "HIGH response (red)",
            "content_area": "full original-image frame",
            "region_source": "full_frame",
            "region_pixels": [0, 0, 64, 80],
            "region_normalized_1000": [0, 0, 1000, 1000],
        }},
    }
    from state_machine.evidence_tokenizer import EvidenceTokenizer as Tokenizer
    visible = Tokenizer.model_visible(token)
    item = visible["artifact_layout"]["anomaly_map"]
    assert item["inspection_region_source"] == "full_frame"
    assert item["region_normalized_1000"] is None
    assert "context view" in item["note"]
