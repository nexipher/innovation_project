"""
The score-semantics registry: one definition of what each expert's number means.

Stage 3 (expert evaluation) reads it to interpret a raw score; stage 4 (evidence
checks) reads it to judge whether the model's prose matches the number it was
given.  Two copies would drift the way v1's did — the audit and the runtime
disagreed about which experts meant "AI-generated" — so there is one file
(`experts/score_semantics.json`) and this loader is the only way in.

What it deliberately keeps apart:

  * **raw score vs calibrated probability.**  `is_calibrated_probability` is
    false for every external discriminator: a sigmoid output is not a
    probability this project fitted, and saying otherwise is how the v1 tables
    ended up being read as ground truth.
  * **direction.**  `higher_is_fake` / `higher_is_real` is a property of the
    expert, verified once; the evidence check must not re-derive it from
    whatever the model wrote.
  * **measurement scope.**  UnivFD centre-crops; the others measure the whole
    image.  A token that says "global" while the expert cropped is the
    `global_evidence_localised` defect, and this field is what makes it
    detectable rather than a matter of opinion.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from config import PROJECT_ROOT

REGISTRY_PATH = os.path.join(PROJECT_ROOT, "experts", "score_semantics.json")

SCHEMA = 1
REQUIRED_FIELDS = ("display_name", "preprocess", "raw_score", "threshold", "status")
DIRECTIONS = ("higher_is_fake", "higher_is_real", "none")
_PREPROCESS_KEYS = ("name", "detail", "measurement_scope")
SCOPES = ("global", "center_crop", "region")


class SemanticsError(ValueError):
    """The registry is malformed — a loud failure, not a default."""


_cache: Optional[dict] = None


def load(path: str = REGISTRY_PATH, use_cache: bool = True) -> dict:
    """The registry, validated.  Raises SemanticsError on a broken file."""
    global _cache
    if use_cache and _cache is not None and path == REGISTRY_PATH:
        return _cache
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise SemanticsError(f"分数语义登记表读不出来：{path}（{exc}）") from exc
    _validate(data, path)
    if path == REGISTRY_PATH:
        _cache = data
    return data


def _validate(data: dict, path: str) -> None:
    if data.get("schema") != SCHEMA:
        raise SemanticsError(f"登记表 schema 应为 {SCHEMA}，实际 {data.get('schema')!r}")
    experts = data.get("experts")
    if not isinstance(experts, dict) or not experts:
        raise SemanticsError("登记表里没有 experts")
    for name, entry in experts.items():
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise SemanticsError(f"{name} 缺少字段：{missing}")
        preprocess = entry["preprocess"]
        missing = [key for key in _PREPROCESS_KEYS if key not in preprocess]
        if missing:
            raise SemanticsError(f"{name}.preprocess 缺少：{missing}")
        if preprocess["measurement_scope"] not in SCOPES:
            raise SemanticsError(
                f"{name}.preprocess.measurement_scope 必须是 {SCOPES} 之一（枚举，不是散文），"
                f"实际 {preprocess['measurement_scope']!r}")
        score = entry["raw_score"]
        if score.get("direction") not in DIRECTIONS:
            raise SemanticsError(
                f"{name}.raw_score.direction 必须是 {DIRECTIONS} 之一，"
                f"实际 {score.get('direction')!r}")
        if "is_calibrated_probability" not in score:
            raise SemanticsError(
                f"{name}.raw_score 必须写明 is_calibrated_probability："
                "sigmoid 分数与本地校准概率是两件事")
        rng = score.get("range")
        if not (isinstance(rng, list) and len(rng) == 2):
            raise SemanticsError(f"{name}.raw_score.range 应为 [low, high]")
        if not score.get("field"):
            raise SemanticsError(f"{name}.raw_score.field 未写明（token 里的哪个字段）")
        if "value" not in entry["threshold"] or "chosen_on" not in entry["threshold"]:
            raise SemanticsError(f"{name}.threshold 必须写明 value 与 chosen_on")


def entry(source_name: str, variant: Optional[str] = None) -> dict:
    """
    One expert's record, by its `source_name` (the token's `source` field).

    `variant` selects between checkpoints of the same expert (NPR has two);
    asking for a variant that does not exist raises rather than silently
    returning the other one.
    """
    experts = load()["experts"]
    if source_name not in experts:
        known = ", ".join(sorted(experts))
        raise SemanticsError(f"登记表里没有专家 {source_name!r}；已知：{known}")
    record = experts[source_name]
    if variant is not None and record.get("variant") != variant:
        raise SemanticsError(
            f"{source_name} 的 variant 是 {record.get('variant')!r}，不是 {variant!r}")
    return record


def direction_of(source_name: str) -> str:
    return entry(source_name)["raw_score"]["direction"]


def measurement_scope_of(source_name: str) -> str:
    return entry(source_name)["preprocess"]["measurement_scope"]


def is_calibrated_probability(source_name: str) -> bool:
    return bool(entry(source_name)["raw_score"]["is_calibrated_probability"])


def checkpoints(source_name: str) -> List[dict]:
    """Every downloadable checkpoint of an expert (empty for in-repo experts)."""
    weights = entry(source_name).get("weights") or {}
    listed = weights.get("checkpoints")
    if listed:
        return list(listed)
    if weights.get("file"):
        return [{**weights, "id": entry(source_name).get("variant")}]
    return []


def summary() -> str:
    """A readable table — what a human checks before trusting an evaluation."""
    lines = ["| 专家 | 变体 | 方向 | 测量范围 | 已校准概率 | 许可 | 评测状态 |",
             "|------|------|------|----------|-----------|------|----------|"]
    for name, record in sorted(load()["experts"].items()):
        score = record["raw_score"]
        weights = record.get("weights") or {}
        lines.append(
            f"| `{name}` | {record.get('variant')} | {score['direction']} | "
            f"{record['preprocess']['measurement_scope']} | "
            f"{'是' if score['is_calibrated_probability'] else '否'} | "
            f"{weights.get('licence', '—')} | {record['status'].get('evaluated', '—')} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summary())
