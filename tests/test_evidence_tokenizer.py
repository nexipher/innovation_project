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
