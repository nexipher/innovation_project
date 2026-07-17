"""Unit tests for the XML tag parser."""

import pytest
from utils.parser import Parser


class TestPlanning:
    def test_extract_planning(self):
        text = "<planning>\nSuspected Region: [100,200,300,400]\n</planning>"
        result = Parser.parse_planning(text)
        assert result is not None
        assert "Suspected Region" in result

    def test_no_planning(self):
        assert Parser.parse_planning("no tags here") is None

    def test_planning_with_extra_text(self):
        text = "prefix <planning>hello world</planning> suffix"
        assert Parser.parse_planning(text) == "hello world"


class TestCallExtraction:
    def test_extract_freq(self):
        text = "<call_freq>[100, 200, 300, 400]</call_freq>"
        calls = Parser.parse_call("freq", text)
        assert calls == [[100, 200, 300, 400]]

    def test_extract_noise(self):
        text = "<call_noise> [50, 60, 70, 80] </call_noise>"
        calls = Parser.parse_call("noise", text)
        assert calls == [[50, 60, 70, 80]]

    def test_extract_jpeg(self):
        text = "<call_jpeg>[0,0,1000,1000]</call_jpeg>"
        calls = Parser.parse_call("jpeg", text)
        assert calls == [[0, 0, 1000, 1000]]

    def test_extract_all_calls(self):
        text = "<call_freq>[100,200,300,400]</call_freq> and <call_noise>[50,60,70,80]</call_noise>"
        calls = Parser.extract_all_calls(text)
        assert len(calls) == 2
        assert ("freq", [100, 200, 300, 400]) in calls
        assert ("noise", [50, 60, 70, 80]) in calls

    def test_no_calls(self):
        assert Parser.extract_all_calls("no calls here") == []

    def test_has_call(self):
        assert Parser.has_call("<call_freq>[1,2,3,4]</call_freq>")
        assert not Parser.has_call("nothing")

    def test_call_without_brackets(self):
        text = "<call_freq>100, 200, 300, 400</call_freq>"
        calls = Parser.parse_call("freq", text)
        assert calls == [[100, 200, 300, 400]]


class TestVerdict:
    def test_extract_verdict(self):
        text = '<verdict>\n{"verdict": "Fake", "confidence": 0.92}\n</verdict>'
        v = Parser.parse_verdict(text)
        assert v == {"verdict": "Fake", "confidence": 0.92}

    def test_no_verdict(self):
        assert Parser.parse_verdict("no verdict") is None

    def test_has_verdict(self):
        assert Parser.has_verdict("<verdict>x</verdict>")
        assert not Parser.has_verdict("nothing")


class TestReasoning:
    def test_extract_reasoning(self):
        text = "<reasoning>\n物理-语义一致性校验通过\n</reasoning>"
        r = Parser.parse_reasoning(text)
        assert "物理-语义一致性校验" in r

    def test_no_reasoning(self):
        assert Parser.parse_reasoning("no tags") is None


class TestValidation:
    def test_valid_structure(self):
        text = "<planning>...</planning><call_freq>[1,2,3,4]</call_freq>"
        valid, msg = Parser.validate_tag_structure(text)
        assert valid

    def test_invalid_no_tags(self):
        valid, msg = Parser.validate_tag_structure("just random text")
        assert not valid
        assert "planning" in msg.lower() or "reasoning" in msg.lower()

    def test_reasoning_with_verdict_valid(self):
        text = "<reasoning>ok</reasoning><verdict>{}</verdict>"
        valid, _ = Parser.validate_tag_structure(text)
        assert valid


class TestFallbackBbox:
    def test_extract_bbox(self):
        bbox = Parser.extract_bbox_from_text("text [100, 200, 300, 400] more")
        assert bbox == [100, 200, 300, 400]

    def test_no_bbox(self):
        assert Parser.extract_bbox_from_text("no brackets") is None

    def test_bbox_out_of_range(self):
        bbox = Parser.extract_bbox_from_text("[0, 0, 2000, 2000]")
        assert bbox is not None  # returns as-is, downstream clip handles it
