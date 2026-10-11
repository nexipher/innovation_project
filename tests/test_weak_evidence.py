"""
Tests for the weak-evidence check (G7 entry 4).

The check exists because 30 of 34 near-chance tokens in the acceptance run were
quoted with a direction attached.  What it must never do is flag the sentence
we want the model to write — "this is near chance and settles nothing" — so
both halves are pinned here.
"""

import pytest

from utils.weak_evidence import WEAK_HIGH, WEAK_LOW, flags, summarise, weak_tokens


def _token(p_fake, evidence_id="E-1", source="probe_dino_expert"):
    return {"evidence_id": evidence_id, "source": source,
            "calibrated_likelihood": {"Real": round(1 - p_fake, 3), "Fake": p_fake}}


def _sample(text, tokens, verdict=None):
    return {
        "id": "s1",
        "evidence_chain": tokens,
        "conversations": [{"from": "gpt", "value": text}],
        "final_verdict": verdict or {"verdict": "Uncertain", "confidence": 0.5},
    }


class TestTheBand:
    def test_the_middle_of_the_band_is_weak(self):
        assert weak_tokens(_sample("", [_token(0.52)]))
        assert WEAK_LOW <= 0.52 <= WEAK_HIGH

    def test_the_edges_count_as_weak(self):
        assert weak_tokens(_sample("", [_token(WEAK_LOW), _token(WEAK_HIGH)]))

    def test_a_clear_measurement_is_not_weak(self):
        assert weak_tokens(_sample("", [_token(0.97), _token(0.02)])) == []

    def test_an_uncalibrated_token_is_not_weak_it_is_absent(self):
        token = {"evidence_id": "E-x", "source": "probe_dino_expert"}
        assert weak_tokens({"evidence_chain": [token]}) == []


class TestWhatGetsFlagged:
    def test_promoting_a_near_chance_token_is_flagged(self):
        sample = _sample("PROBE 鉴别器的判别分数明确表明图像是伪造的。", [_token(0.52)])
        found = flags(sample)
        assert len(found) == 1
        assert found[0]["word"] == "明确表明" and found[0]["p_fake"] == 0.52

    def test_calling_it_decisive_is_flagged(self):
        sample = _sample("该判别分数是决定性证据。", [_token(0.45)])
        assert flags(sample)[0]["word"] == "决定性"

    def test_the_honest_reading_is_not_flagged(self):
        sample = _sample("PROBE 的判别分数接近随机（P(Fake)=0.52），略偏向伪造，"
                         "但不足以单独支撑判定。", [_token(0.52)])
        assert flags(sample) == []

    def test_denying_a_strong_claim_is_not_flagged(self):
        """'This does not prove anything' is the sentence we are training for."""
        sample = _sample("该分数不构成决定性证据，只能作为弱线索。", [_token(0.5)])
        assert flags(sample) == []

    def test_a_strong_word_about_a_clear_token_is_not_our_business(self):
        sample = _sample("PROBE 鉴别器的判别分数明确表明图像是伪造的。", [_token(0.99)])
        assert flags(sample) == []

    def test_a_strong_word_about_another_expert_is_not_flagged(self):
        sample = _sample("噪声残留水平明确表明这是相机拍摄。",
                         [_token(0.52), _token(0.11, "E-2", "noise_expert")])
        assert flags(sample) == []

    def test_confidence_resting_on_a_weak_token_is_flagged(self):
        sample = _sample("结论如下。", [_token(0.52)],
                         verdict={"verdict": "Fake", "confidence": 0.95,
                                  "primary_evidence": ["E-1"]})
        found = flags(sample)
        assert any(entry["word"] == "primary_evidence" for entry in found)

    def test_a_confident_verdict_without_weak_support_is_fine(self):
        sample = _sample("结论如下。", [_token(0.52)],
                         verdict={"verdict": "Fake", "confidence": 0.95,
                                  "primary_evidence": ["E-2"]})
        assert flags(sample) == []


class TestSummary:
    def test_the_summary_counts_samples_and_sources(self):
        samples = [
            _sample("判别分数明确表明是伪造。", [_token(0.52)]),
            _sample("该分数接近随机。", [_token(0.52)]),
        ]
        samples[0]["id"], samples[1]["id"] = "a", "b"
        report = summarise(samples)
        assert report["samples"] == 2
        assert report["samples_with_weak_tokens"] == 2
        assert report["flagged"] == 1
        assert report["details"]["a"]
