"""Unit tests for Evidence Tokenizer."""

from state_machine.evidence_tokenizer import EvidenceTokenizer


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
