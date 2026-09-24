"""
Experiment configuration fingerprint (plan.md §4.10 G3-d).

A resumable evaluation must be able to tell "the same experiment, continued"
from "a different experiment that happens to share a mode and a sample size".
The resume key used to be `(mode, per_cell)` alone, so a report produced before
the G2-e polarity fix would be silently reused by a run made after it — the
numbers would look unchanged because they *were* the old numbers.

The fingerprint covers everything that changes what a session measures:

  * the expert registry: each source name, its class and its metric polarity;
  * the system prompts (forensic and no-tool baseline) by content hash;
  * the reliability table the tokens are calibrated against;
  * the rectifier/tokenizer/policy versions;
  * the git commit.

Comparing a stored fingerprint against the live one is exact: any difference
means a cold start, and the caller reports which component differs.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Dict, Optional

from config import HALTING_POLICY, PROJECT_ROOT

# Bumped when the component changes in a way that alters session records.
RECTIFIER_VERSION = "g3b"
TOKENIZER_VERSION = "g3a"

RELIABILITY_TABLE_PATH = os.path.join(PROJECT_ROOT, "calibration", "reliability_table.json")


def git_commit() -> str:
    """Short HEAD hash, or 'unknown' outside a repository checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _file_digest(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()[:16]
    except OSError:
        return None


def prompt_digest() -> str:
    """Hash of the system prompts the model is run under."""
    from mllm.message_builder import BASELINE_SYSTEM_PROMPT, FORENSIC_SYSTEM_PROMPT

    payload = json.dumps(
        {"forensic": FORENSIC_SYSTEM_PROMPT, "baseline": BASELINE_SYSTEM_PROMPT},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def expert_specs(experts: Dict[str, object]) -> Dict[str, dict]:
    """
    Identify each registered expert by what changes its measurements.

    The polarity matters as much as the class: the same expert with an
    inverted reading is a different instrument.
    """
    return {
        name: {
            "class": type(expert).__name__,
            "polarity": int(getattr(expert, "metric_polarity", 1)),
            "source_name": str(getattr(expert, "source_name", name)),
        }
        for name, expert in sorted(experts.items())
    }


def compute(experts: Dict[str, object]) -> dict:
    """Assemble the fingerprint for one evaluation configuration."""
    components = {
        "experts": expert_specs(experts),
        "prompts": prompt_digest(),
        "reliability_table": _file_digest(RELIABILITY_TABLE_PATH),
        "rectifier": RECTIFIER_VERSION,
        "tokenizer": TOKENIZER_VERSION,
        "halting_policy": HALTING_POLICY,
        "git_commit": git_commit(),
    }
    components["digest"] = digest(components)
    return components


def digest(fingerprint: dict) -> str:
    """Stable short digest over every component except the digest itself."""
    payload = {key: value for key, value in fingerprint.items() if key != "digest"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def differences(stored: Optional[dict], live: dict) -> list:
    """
    Which components differ, for a caller that wants to report why it is
    starting over rather than resuming.
    """
    if not stored:
        return ["no fingerprint on record"]
    names = set(stored) | set(live)
    return sorted(
        name for name in names
        if name != "digest" and stored.get(name) != live.get(name)
    )
