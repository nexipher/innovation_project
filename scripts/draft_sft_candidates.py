"""Turn the reviewed case content into protocol-shaped assistant scripts.

This is a drafting/validation step only: it does not run an expert or write a
page.  It refuses a case whose verdict fields contradict each other, a Tampered
case without a supported region, a non-null certainty for Uncertain, or a
forbidden "mutual verification" phrase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/root/autodl-tmp/innovation_project")
sys.path.insert(0, str(ROOT))

from mllm.message_builder import build_forensic_prompt  # noqa: E402
from scripts.sft_candidate_cases import CASES  # noqa: E402

DEFAULT_OUT = ROOT / "sft_data/sft_candidates_20261127"
FORBIDDEN = ("交叉验证", "相互验证", "共同支持", "相互支持")


def planning_text(entry: dict, call: str, first: bool) -> str:
    target = entry["target"]
    if call == "probe":
        # Confirmed rule: the PROBE call reason states the whole-image origin
        # question only; it must not be phrased as a local-tampering check.
        target = ("probe — 整图来源尚未测量；PROBE 是整图来源的判别器。"
                  if first else
                  "probe — 整图来源仍未测量；PROBE 是整图来源的判别器，"
                  "补测以完成整图来源判断。")
    return (
        "<planning>\n"
        f"Suspected Region: {entry['region']}\n"
        f"Visual Observation: {entry['visual']}\n"
        f"Suspected Anomalies: {entry['anomalies']}\n"
        f"Expert Target & Hypothesis: {target}\n"
        "</planning>"
    )


def call_text(call: str, region: str) -> str:
    if call == "probe" or region == "none":
        return f"<call_{call}></call_{call}>"
    return f"<call_{call}>{region}</call_{call}>"


def build_outputs(case: dict) -> list[str]:
    outputs = []
    calls = case["call_order"]
    if not calls:
        raise SystemExit("empty call order")
    outputs.append(planning_text(case["first"], calls[0], True) + "\n" +
                   call_text(calls[0], case["first"]["region"]))
    if len(calls) == 2:
        if not case.get("second"):
            raise SystemExit("two calls but no second planning block")
        outputs.append(planning_text(case["second"], calls[1], False) + "\n" +
                       call_text(calls[1], case["second"]["region"]))
    elif len(calls) > 2:
        raise SystemExit("at most two expert calls are supported")
    verdict = case["verdict"]
    outputs.append(
        "<reasoning>\n" + case["reasoning"].strip() + "\n</reasoning>\n"
        "<verdict>\n" + json.dumps(verdict, ensure_ascii=False) + "\n</verdict>")
    return outputs


def validate(case: dict) -> None:
    verdict = case["verdict"]
    whole = verdict.get("whole_image_generation")
    local = verdict.get("local_manipulation")
    overall = verdict.get("verdict")
    certainty = verdict.get("certainty")
    regions = verdict.get("suspicious_regions")
    if whole == "AI-generated" and overall != "Fake":
        raise SystemExit(f"{overall}: AI-generated must be Fake")
    if local == "Tampered" and overall != "Fake":
        raise SystemExit(f"{overall}: Tampered must be Fake")
    if local == "Not applicable" and whole != "AI-generated":
        raise SystemExit("Not applicable requires AI-generated")
    if overall == "Uncertain" and certainty is not None:
        raise SystemExit("Uncertain must have null certainty")
    if overall in ("Real", "Fake") and certainty not in ("High", "Medium", "Low"):
        raise SystemExit(f"{overall} needs non-null certainty")
    if local == "Tampered" and not regions:
        raise SystemExit("Tampered needs a suspicious region")
    if local == "Not tampered" and regions:
        raise SystemExit("Not tampered should not carry a suspicious region")
    for region in regions or []:
        if not region.get("bbox") or len(region["bbox"]) != 4:
            raise SystemExit("bad suspicious region bbox")
        if not region.get("evidence"):
            raise SystemExit("suspicious region has no evidence id")
    text = case["reasoning"] + case["verdict"].get("report", "")
    for phrase in FORBIDDEN:
        if phrase in text:
            raise SystemExit(f"forbidden phrase {phrase!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    out = Path(args.out)
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    ids = [row["sample_id"] for row in manifest["samples"]]
    if set(ids) != set(CASES):
        raise SystemExit(f"case set mismatch: missing={set(ids)-set(CASES)} "
                         f"extra={set(CASES)-set(ids)}")
    prompt = build_forensic_prompt(("probe", "trufor"))
    samples = []
    for row in manifest["samples"]:
        sid = row["sample_id"]
        case = CASES[sid]
        validate(case)
        outputs = build_outputs(case)
        samples.append({
            "sample_id": sid,
            "key_case": row.get("key_case"),
            "key_case_rank": row.get("key_case_rank"),
            "selection_role": row.get("selection_role"),
            "call_order": case["call_order"],
            "outputs": outputs,
            "verdict": case["verdict"],
        })
    payload = {
        "schema": "sft_candidate_scripts/1",
        "assistant_source": "scripted candidate writer (not Qwen output)",
        "system_prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8")).hexdigest(),
        "system_prompt_chars": len(prompt),
        "samples": samples,
    }
    (out / "candidate_scripts.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({
        "out": str(out / "candidate_scripts.json"),
        "samples": len(samples),
        "turns": sum(len(s["outputs"]) for s in samples),
        "call_order": {s["sample_id"]: s["call_order"] for s in samples},
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
