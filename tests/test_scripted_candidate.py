"""The scripted candidate client only cites evidence already delivered."""

from mllm.scripted_candidate import ScriptedCandidateMLLM


def _tool_turn(expert: str, evidence_id: str, value: float) -> dict:
    return {"from": "user", "value": (
        '{"evidence_id": "%s", "expert": "%s", "score": {"value": %s}}'
        % (evidence_id, expert, value))}


def test_placeholders_fill_from_delivered_tokens_only():
    client = ScriptedCandidateMLLM([
        "before {{PROBE_ID}} / {{TRUFOR_ID}} / {{PROBE_SCORE}} / {{TRUFOR_SCORE}}",
    ])
    history = [
        {"from": "user", "value": "<image>"},
        _tool_turn("trufor_expert", "E-aaaaaaaaaa", 0.9),
    ]
    output = client.generate("/tmp/example.jpg", history)
    assert "E-aaaaaaaaaa" in output
    assert "{{PROBE_ID}}" not in output
    assert "{{TRUFOR_ID}}" not in output
    assert "0.9" in output


def test_reset_restarts_the_script():
    client = ScriptedCandidateMLLM(["first", "second"])
    assert client.generate("/tmp/a.jpg", []) == "first"
    client.reset()
    assert client.generate("/tmp/a.jpg", []) == "first"


def test_exhausted_script_is_an_error():
    client = ScriptedCandidateMLLM(["only"])
    client.generate("/tmp/a.jpg", [])
    try:
        client.generate("/tmp/a.jpg", [])
    except RuntimeError as exc:
        assert "exhausted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected RuntimeError")
