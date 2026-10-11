"""
TruFor adapter (G7, plan.md §4.16) — manipulation anomaly, kept global.

What it is
----------
TruFor (CVPR 2023, GRIP-UNINA + Google) fuses an RGB stream with Noiseprint++,
a self-supervised noise fingerprint, and produces three things: a per-pixel
anomaly map (`map`, P(forged) per pixel), a whole-image detection score
(`score` ∈ [0,1]), and a reliability map (`conf`) marking where the
localisation is likely wrong.  `--save-np` also writes Noiseprint++.

How it is run
-------------
Its official environment is python 3.7 / torch 1.11 / opencv 4.4, which is not
ours, so this adapter shells out to `test.py` in the TruFor checkout and reads
the `.npz` it writes.  No second forward pass: `analyze()` runs the model once
and keeps the arrays, `render_artifacts()` paints those same arrays.

Scope (G7-4, decided 2026-10-11)
--------------------------------
The project's task stays *global detection*; L1–L3 is not opened yet.  So the
token's measurement scope is global and the anomaly map is delivered as
(a) aggregate statistics in the metadata and (b) a diagnostic image.  Text may
describe "模型预测的高异常响应区域" — a prediction — but never assert that an
area has been confirmed as tampered.

Licence (G7-5)
--------------
GRIP-UNINA: "informational and non-profit purposes" only.  The weights are not
redistributed with this project; `TRUFOR_HOME` has no default.

Usage
-----
    TRUFOR_HOME=/path/to/TruFor \\
    TRUFOR_PYTHON=/path/to/conda/envs/trufor/bin/python \\
    python -c "from experts.trufor import TruForExpert; ..."
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import tempfile
from typing import Dict, List, Optional

import numpy as np

from experts.external_base import ExternalScoreExpert, MockExternalScoreExpert

# Above this P(forged) a pixel counts towards the "high response" area
# fraction.  TruFor's own camera-ready uses 0.5 for its localisation metrics;
# the number is recorded in the metadata so a later re-analysis can re-derive
# the statistic from the stored map without re-running the model.
HIGH_RESPONSE_THRESHOLD = 0.5


class TruForExpert(ExternalScoreExpert):
    """Whole-image manipulation score + diagnostic maps, run as a subprocess."""

    source_name = "trufor_expert"
    evidence_name = "manipulation_anomaly_score"
    weights_env = "TRUFOR_HOME"
    weights_filename = "pretrained_models/trufor.pth.tar"
    weight_source = "GRIP-UNINA TruFor (CVPR 2023) · TruFor_weights.zip"
    licence_note = ("GRIP-UNINA：仅限信息性与非营利用途；本项目不再分发其代码与权重")
    score_reference = "TruFor 检测头（RGB + Noiseprint++ 融合）"

    experiment = "trufor_ph3"
    timeout_s = 300

    def __init__(self, weights_path: Optional[str] = None, device: str = "cpu",
                 python: Optional[str] = None, save_noiseprint: bool = False,
                 enabled: bool = True):
        super().__init__(weights_path, device, enabled)
        self.python = python or os.environ.get("TRUFOR_PYTHON", "python3")
        self.save_noiseprint = save_noiseprint
        #: arrays from the last analyze(), so render_artifacts costs nothing
        self._last: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    @property
    def home(self) -> str:
        """The TruFor checkout (the directory two levels above the weights)."""
        if not self.weights_path:
            return ""
        return os.path.dirname(os.path.dirname(self.weights_path))

    def script(self) -> str:
        return os.path.join(self.home, "TruFor_train_test", "test.py")

    def unavailable_reason(self) -> str:
        reason = super().unavailable_reason()
        if reason:
            return reason
        if not os.path.exists(self.script()):
            return f"{self.source_name}：未找到 {self.script()}（TRUFOR_HOME 指向仓库根目录）"
        if shutil.which(self.python) is None and not os.path.exists(self.python):
            return f"{self.source_name}：解释器不可用 {self.python}（设置 TRUFOR_PYTHON）"
        return ""

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def command(self, image_path: str, out_dir: str) -> List[str]:
        """The exact CLI, exposed so a test can assert on it without a GPU."""
        command = [
            self.python, "test.py",
            "-in", image_path,
            "-out", out_dir,
            "-exp", self.experiment,
            "-g", "0" if self.device.startswith("cuda") else "-1",
            "TEST.MODEL_FILE", self.weights_path,
        ]
        if self.save_noiseprint:
            command.append("--save_np")
        return command

    def _score(self, img_bgr: np.ndarray):  # pragma: no cover - subprocess path
        from PIL import Image

        with tempfile.TemporaryDirectory(prefix="trufor_") as work:
            image_path = os.path.join(work, "input.png")
            Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1])).save(image_path)
            out_dir = os.path.join(work, "out")
            os.makedirs(out_dir, exist_ok=True)
            completed = subprocess.run(
                self.command(image_path, out_dir),
                cwd=os.path.join(self.home, "TruFor_train_test"),
                capture_output=True, text=True, timeout=self.timeout_s,
            )
            outputs = sorted(glob.glob(os.path.join(out_dir, "**", "*.npz"), recursive=True))
            if not outputs:
                raise RuntimeError(
                    f"{self.source_name}：未产出 npz（退出码 {completed.returncode}）\n"
                    f"stdout 尾部：{completed.stdout[-500:]}\nstderr 尾部：{completed.stderr[-500:]}")
            arrays = dict(np.load(outputs[0], allow_pickle=True))
        return self.result_from_arrays(arrays)

    # ------------------------------------------------------------------
    # Summaries and artifacts (pure numpy — CPU-testable)
    # ------------------------------------------------------------------

    @staticmethod
    def summarise(arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Collapse TruFor's maps into the statistics that go into the token.

        The maps themselves are kept (as artifacts and in the run record), so
        a later threshold change can be applied offline instead of re-running
        the model (G7-2).
        """
        stats: Dict[str, float] = {}
        if "score" in arrays:
            stats["score"] = float(np.asarray(arrays["score"]).reshape(-1)[0])
        anomaly = arrays.get("map")
        if anomaly is not None:
            anomaly = np.asarray(anomaly, dtype=np.float64)
            high = anomaly >= HIGH_RESPONSE_THRESHOLD
            stats.update({
                "anomaly_peak": float(anomaly.max()),
                "anomaly_mean": float(anomaly.mean()),
                "high_response_fraction": float(high.mean()),
                "anomaly_entropy": _entropy(anomaly),
            })
        confidence = arrays.get("conf")
        if confidence is not None:
            confidence = np.asarray(confidence, dtype=np.float64)
            stats.update({
                "confidence_mean": float(confidence.mean()),
                "low_confidence_fraction": float((confidence < 0.5).mean()),
            })
            if anomaly is not None:
                # What the anomaly map is worth, weighted by where TruFor says
                # its own localisation is trustworthy.
                stats["confidence_weighted_anomaly"] = float(
                    (anomaly * confidence).mean())
        return stats

    def result_from_arrays(self, arrays: Dict[str, np.ndarray]):
        """Token payload from one npz — the half that needs no model."""
        stats = self.summarise(arrays)
        self._last = arrays
        score = float(stats.get("score", 0.0))
        size = arrays.get("imgsize")
        preprocess = "原图直接前向（TruFor 自行缩放），输出 map/conf/score"
        if size is not None:
            preprocess += f"，imgsize={tuple(int(v) for v in np.asarray(size).reshape(-1))}"
        return self._result(
            score, preprocess=preprocess,
            extra={"high_response_threshold": HIGH_RESPONSE_THRESHOLD,
                   "has_noiseprint": "np++" in arrays,
                   "stats": stats},
        )

    def render_artifacts(self, img_patch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Paint the maps from the *last* analysis — no second forward pass.

        Two products, both diagnostic: the anomaly response and the confidence
        map.  They are colourised (JET) so the composition of the evidence
        matches the other experts' artifacts, and they carry no claim: the
        wording rule (G7-4) is enforced on the text, not here.
        """
        if not self._last:
            return {}
        import cv2

        artifacts: Dict[str, np.ndarray] = {}
        for key, name in (("map", "anomaly_map"), ("conf", "confidence_map")):
            array = self._last.get(key)
            if array is None:
                continue
            scaled = np.clip(np.asarray(array, dtype=np.float64) * 255.0, 0, 255).astype(np.uint8)
            artifacts[name] = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        noiseprint = self._last.get("np++")
        if noiseprint is not None:
            magnitude = np.abs(np.asarray(noiseprint, dtype=np.float64))
            peak = magnitude.max() or 1.0
            scaled = np.clip(magnitude / peak * 255.0, 0, 255).astype(np.uint8)
            artifacts["noiseprint_pp"] = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        return artifacts


def _entropy(array: np.ndarray, bins: int = 32) -> float:
    """Shannon entropy of the (normalised) response distribution, in bits."""
    flat = np.asarray(array, dtype=np.float64).reshape(-1)
    if flat.size == 0 or not np.isfinite(flat).all():
        return 0.0
    low, high = float(flat.min()), float(flat.max())
    if high <= low:
        return 0.0
    histogram, _ = np.histogram((flat - low) / (high - low), bins=bins,
                                range=(0.0, 1.0), density=False)
    probabilities = histogram / histogram.sum()
    probabilities = probabilities[probabilities > 0]
    return float(-(probabilities * np.log2(probabilities)).sum())


class MockTruForExpert(TruForExpert):
    """
    Deterministic TruFor stand-in: same class as production, minus the model.

    It subclasses the real adapter on purpose — `analyze` fabricates the
    arrays a run would produce and hands them to the same `result_from_arrays`
    and `render_artifacts` the GPU path uses, so a mock run cannot drift from
    the code it stands in for.
    """

    weight_source = "mock"
    licence_note = "mock（不涉及上游权重）"
    score_reference = "TruFor 检测头（Mock）"

    def __init__(self, score: Optional[float] = None, shape=(64, 64), **kwargs):
        kwargs.setdefault("weights_path", "/dev/null")
        super().__init__(**kwargs)
        self._fixed_score = score
        self._shape = shape
        self._last = None

    @classmethod
    def _resolve_weights_path(cls) -> str:
        return "/dev/null"

    def unavailable_reason(self) -> str:
        return "" if self.enabled else f"{self.source_name} 已停用"

    def analyze(self, img_patch: np.ndarray):
        height, width = self._shape
        rows = np.linspace(0.0, 1.0, height).reshape(-1, 1)
        arrays = {
            "score": np.array(0.5 if self._fixed_score is None else self._fixed_score),
            "map": np.repeat(rows, width, axis=1),
            "conf": np.full((height, width), 0.8),
            "imgsize": np.array([height, width]),
        }
        return self.result_from_arrays(arrays)
