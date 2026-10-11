"""
Shared scaffolding for experts that wrap an external model (G7, plan.md §4.16).

PROBE-DINOv2 and TruFor are not ours: their weights live outside this
repository, their licences are not ours to grant, and neither metric means the
same thing as the statistical experts' strengths.  What this module provides
is the part that *is* ours —

  * the contract the state machine consumes (`ExpertResult`, `render_artifacts`);
  * a provenance record for the weights (source, size, sha256) so a run can be
    reproduced and, if asked, audited;
  * a text builder that cannot invent physics for a black-box score: an
    external detector's number is a discriminator output, not a measurement of
    a physical trace, and the three-layer wording has to say so;
  * an "available" check, so a checkout without the external dependencies keeps
    running and its tests keep passing (G7-5).

Nothing here imports torch, transformers or the external repositories.  The
subclasses import them lazily, in `_load`, and only when a real analysis is
asked for.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from experts.base import BaseExpert, ExpertResult

# A score we have not calibrated carries no direction.  Saying "Uncertain" is
# not a placeholder: the pipeline's own rule is that an uncalibrated token may
# not move the posterior (G2-e), and this is how that rule is expressed at the
# expert boundary.
UNCALIBRATED_SUPPORT = "Uncertain"
UNCALIBRATED_STANCE = (
    "该分数尚未在本地校准集上建立条件可靠性与适用条件，因此不作为方向性证据："
    "它只作为原始测量被记录，方向待 G7-1 校准后确定。"
)


def sha256_file(path: str, cache: bool = True) -> str:
    """
    Hash a weights file, caching the digest beside it.

    The checkpoints are 1.2 GB (PROBE) and 249 MB (TruFor); hashing them on
    every construction would dominate the smoke tests' runtime, and the
    provenance only needs to change when the file does.
    """
    sidecar = f"{path}.sha256"
    if cache and os.path.exists(sidecar):
        try:
            with open(sidecar, encoding="utf-8") as handle:
                recorded = handle.read().split()[0]
            if recorded:
                return recorded
        except OSError:
            pass
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if cache:
        try:
            with open(sidecar, "w", encoding="utf-8") as handle:
                handle.write(f"{value}  {os.path.basename(path)}\n")
        except OSError:
            pass
    return value


@dataclass
class WeightProvenance:
    """Where a checkpoint came from, and what it is."""

    path: str
    source: str = ""
    licence_note: str = ""
    sha256: str = ""
    size_bytes: int = 0

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "source": self.source,
            "licence": self.licence_note,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    def describe(self) -> str:
        return (f"{os.path.basename(self.path)} "
                f"({self.size_bytes / 1e6:.1f} MB, sha256 {self.sha256[:16]})\n"
                f"    来源：{self.source}\n    许可：{self.licence_note}")


def tile_positions(height: int, width: int, tile: int,
                   stride: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
    """
    Non-overlapping `tile`-sized windows covering an image, edges included.

    PROBE scores 336×336 crops and averages them, so a large image needs the
    same tiling to produce one number; the last window on each axis is pulled
    back to the edge rather than dropped.
    """
    stride = stride or tile
    ys = list(range(0, max(height - tile, 0) + 1, stride)) or [0]
    xs = list(range(0, max(width - tile, 0) + 1, stride)) or [0]
    if ys[-1] + tile < height:
        ys.append(max(height - tile, 0))
    if xs[-1] + tile < width:
        xs.append(max(width - tile, 0))
    return [(x, y, tile, tile) for y in ys for x in xs]


def score_phenomenon(score: float, preprocess: str) -> str:
    """Layer 1 of the three: what was measured, on what input."""
    return (f"整图判别分数 {score:.4f}（外部判别器的原始输出，未经本地校准）。"
            f"输入预处理：{preprocess}。")


def score_reasoning(score: float, reference: str) -> str:
    """
    Layer 2: what the number is, and what it is not.

    The layer the review demanded: an external detector's output is a
    *learned discriminator's* response.  It can be described as a prediction,
    never as a physical trace the model found.
    """
    return (f"该分数是外部判别器对整幅图像的预测输出（{reference}），"
            f"用于与本地校准集比较，衡量的是与训练分布的接近程度；"
            "它不是任何物理量的测量，也不能用来解释模型为何这样判断。")


def score_counter_explanation(resolution: str = "") -> str:
    """Layer 3: the benign explanations, and the applicability caveat."""
    body = ("外部判别器的分数随分辨率、重编码与训练分布漂移而变化；"
            "同分布之外的图像（未见生成器、强后处理）可能得到误导性分数，"
            "因此只有在本地校准集上按格式/内容分层验证过的条件下才可作为证据。")
    if resolution:
        body += f" 本项目当前条件：{resolution}。"
    return body


class ExternalScoreExpert(BaseExpert):
    """
    Base for experts whose number comes from a model we did not train.

    Subclasses set `source_name`, `evidence_name`, the environment variable
    that points at the weights, and implement `_load()` + `_score()`.
    """

    evidence_name = "external_detector_score"
    #: environment variable that locates the checkpoint
    weights_env = ""
    #: default file name under that location, when the env var is a directory
    weights_filename = ""
    weight_source = ""
    licence_note = ""
    #: what the score means, for the reasoning text
    score_reference = ""

    def __init__(self, weights_path: Optional[str] = None,
                 device: str = "cpu", enabled: bool = True):
        self.weights_path = weights_path or self._resolve_weights_path()
        self.device = device
        self.enabled = enabled
        self._model = None
        self._provenance: Optional[WeightProvenance] = None

    # ------------------------------------------------------------------
    # Availability and provenance
    # ------------------------------------------------------------------

    @classmethod
    def _resolve_weights_path(cls) -> str:
        if not cls.weights_env:
            return ""
        raw = os.environ.get(cls.weights_env, "")
        if not raw:
            return ""
        if cls.weights_filename and os.path.isdir(raw):
            return os.path.join(raw, cls.weights_filename)
        return raw

    def unavailable_reason(self) -> str:
        """Why this expert cannot run, or "" when it can (no heavy imports)."""
        if not self.enabled:
            return f"{self.source_name} 已停用"
        if not self.weights_path:
            return (f"{self.source_name}：未配置权重路径"
                    f"（设置环境变量 {self.weights_env}）")
        if not os.path.exists(self.weights_path):
            return f"{self.source_name}：权重文件不存在 {self.weights_path}"
        missing = self._missing_dependency()
        if missing:
            return f"{self.source_name}：缺少依赖 {missing}"
        return ""

    def available(self) -> bool:
        return self.unavailable_reason() == ""

    @staticmethod
    def _missing_dependency() -> str:
        return ""

    def provenance(self) -> WeightProvenance:
        if self._provenance is None:
            path = self.weights_path
            self._provenance = WeightProvenance(
                path=path,
                source=self.weight_source,
                licence_note=self.licence_note,
                sha256=sha256_file(path) if path and os.path.exists(path) else "",
                size_bytes=os.path.getsize(path) if path and os.path.exists(path) else 0,
            )
        return self._provenance

    # ------------------------------------------------------------------
    # The contract
    # ------------------------------------------------------------------

    def analyze(self, img_patch: np.ndarray) -> ExpertResult:
        reason = self.unavailable_reason()
        if reason:
            raise RuntimeError(reason)
        result = self._score(img_patch)
        return result

    def _score(self, img_patch: np.ndarray) -> ExpertResult:      # pragma: no cover
        raise NotImplementedError

    def _result(self, score: float, preprocess: str,
                extra: Optional[Dict] = None) -> ExpertResult:
        """
        Assemble the token payload.

        `BaseExpert._build_result` reads `counter_explanation` off the
        instance (not from an argument) and derives `region` from the bbox, so
        both are set here rather than passed — the expert measures the whole
        image and its caveat is a property of the expert, not of one call.
        """
        provenance = self.provenance()
        metadata = {
            "preprocess": preprocess,
            "weights_sha256": provenance.sha256,
            "weights_source": provenance.source,
            "licence": provenance.licence_note,
            "calibrated": False,
        }
        metadata.update(extra or {})
        self.counter_explanation = score_counter_explanation()
        return self._build_result(
            evidence_name=self.evidence_name,
            phenomenon=score_phenomenon(score, preprocess),
            reasoning=score_reasoning(score, self.score_reference),
            strength=float(np.clip(score, 0.0, 1.0)),
            support=UNCALIBRATED_SUPPORT,
            interpretation_text=UNCALIBRATED_STANCE,
            raw_metric=float(score),
            **metadata,
        )


class MockExternalScoreExpert(ExternalScoreExpert):
    """
    Deterministic stand-in for CPU tests and dry runs.

    Same `source_name` as the real expert, so a pipeline exercised with the
    mock behaves like one exercised with the model — which is the point of the
    project's mock rule (agent.md §3.2.2).  The score is a stable function of
    the image content, so a test can assert on it.
    """

    def __init__(self, score: Optional[float] = None, **kwargs):
        kwargs.setdefault("weights_path", "/dev/null")   # nothing to load
        super().__init__(**kwargs)
        self._fixed_score = score

    @classmethod
    def _resolve_weights_path(cls) -> str:
        return "/dev/null"

    def unavailable_reason(self) -> str:
        return "" if self.enabled else f"{self.source_name} 已停用"

    def _score(self, img_patch: np.ndarray) -> ExpertResult:
        if self._fixed_score is not None:
            score = float(self._fixed_score)
        else:
            digest = hashlib.sha256(np.ascontiguousarray(img_patch).tobytes()).digest()
            score = int.from_bytes(digest[:4], "big") / 2 ** 32
        return self._result(score, preprocess="mock（确定性哈希）",
                            extra={"mock": True})
