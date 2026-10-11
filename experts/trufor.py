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
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

from config import PROJECT_ROOT
from experts.external_base import ExternalScoreExpert

# Above this P(forged) a pixel counts towards the "high response" area
# fraction.  TruFor's own camera-ready uses 0.5 for its localisation metrics;
# the number is recorded in the metadata so a later re-analysis can re-derive
# the statistic from the stored map without re-running the model.
HIGH_RESPONSE_THRESHOLD = 0.5

# Bumped when the way an image is fed to TruFor changes: the cache key
# includes it, so an old cached result can never be mistaken for one produced
# by the current recipe.
RECIPE_VERSION = "trufor-v1"

DEFAULT_RESULTS_DIR = os.path.join(PROJECT_ROOT, "calibration", "trufor_raw")


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
                 enabled: bool = True, results_dir: Optional[str] = None,
                 cache: bool = True):
        super().__init__(weights_path, device, enabled)
        self.python = python or os.environ.get("TRUFOR_PYTHON", "python3")
        self.save_noiseprint = save_noiseprint
        # Raw products are kept, not discarded with the temporary directory:
        # the score, the maps and the run's metadata go to a fixed directory so
        # a later threshold, calibration or fusion change can be re-derived
        # offline instead of re-running the model (G7-2).
        self.results_dir = (results_dir
                            or os.environ.get("TRUFOR_RESULTS")
                            or DEFAULT_RESULTS_DIR)
        self.cache = cache
        #: arrays from the last analyze(), so render_artifacts costs nothing
        self._last: Optional[Dict[str, np.ndarray]] = None
        #: cold start is the first call in this process: it pays for the
        #: interpreter, the imports and the checkpoint.  The batch decision
        #: (one subprocess per image vs. a resident worker) rests on these.
        self._runs = 0
        self.timings: List[dict] = []

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

    # ------------------------------------------------------------------
    # Raw-product cache (G7-2)
    # ------------------------------------------------------------------

    def cache_key(self, img_bgr: np.ndarray) -> str:
        """
        Key a result by everything that could change it.

        Image bytes plus the weights hash plus the recipe version: pointing the
        adapter at different weights, or changing how the image is fed, must
        not silently reuse a number produced under the old setup.
        """
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(img_bgr).tobytes())
        digest.update(b"|")
        digest.update(self.provenance().sha256.encode("ascii"))
        digest.update(f"|{RECIPE_VERSION}".encode("ascii"))
        digest.update(f"|{self.experiment}|{HIGH_RESPONSE_THRESHOLD}".encode("ascii"))
        return digest.hexdigest()[:24]

    def result_path(self, key: str) -> str:
        return os.path.join(self.results_dir, key, "result.npz")

    def meta_path(self, key: str) -> str:
        return os.path.join(self.results_dir, key, "meta.json")

    def cached_result(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        path = self.result_path(key)
        if not (self.cache and os.path.exists(path)):
            return None
        return dict(np.load(path, allow_pickle=True))

    def persist(self, key: str, arrays: Dict[str, np.ndarray], meta: dict) -> str:
        directory = os.path.join(self.results_dir, key)
        os.makedirs(directory, exist_ok=True)
        np.savez(self.result_path(key), **arrays)
        with open(self.meta_path(key), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=1)
        return directory

    def _score(self, img_bgr: np.ndarray):  # pragma: no cover - subprocess path
        from PIL import Image

        key = self.cache_key(img_bgr)
        cached = self.cached_result(key)
        if cached is not None:
            self._last = cached
            self.timings.append({"key": key, "cold_start": False,
                                 "elapsed_s": 0.0, "cached": True})
            return self.result_from_arrays(cached)

        started = time.perf_counter()
        cold_start = self._runs == 0
        with tempfile.TemporaryDirectory(prefix="trufor_") as work:
            image_path = os.path.join(work, "input.png")
            Image.fromarray(np.ascontiguousarray(img_bgr[:, :, ::-1])).save(image_path)
            out_dir = os.path.join(work, "out")
            os.makedirs(out_dir, exist_ok=True)
            returncode, stdout, stderr, npz_path = self._run_tool(image_path, out_dir)
            if not npz_path:
                raise RuntimeError(
                    f"{self.source_name}：未产出 npz（退出码 {returncode}）\n"
                    f"stdout 尾部：{stdout[-500:]}\nstderr 尾部：{stderr[-500:]}")
            arrays = dict(np.load(npz_path, allow_pickle=True))

        elapsed = time.perf_counter() - started
        self._runs += 1
        self.timings.append({"key": key, "cold_start": cold_start,
                             "elapsed_s": round(elapsed, 2), "cached": False})
        self.persist(key, arrays, {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "image_sha256": hashlib.sha256(
                np.ascontiguousarray(img_bgr).tobytes()).hexdigest(),
            "weights": self.provenance().as_dict(),
            "experiment": self.experiment,
            "recipe_version": RECIPE_VERSION,
            "preprocess": "原图 PNG 落盘 → TruFor test.py 直接前向",
            "high_response_threshold": HIGH_RESPONSE_THRESHOLD,
            "cold_start": cold_start,
            "elapsed_s": round(elapsed, 2),
            "subprocess": {"returncode": returncode,
                           "stdout_tail": stdout[-400:],
                           "stderr_tail": stderr[-400:]},
            "stats": self.summarise(arrays),
            "npz_keys": sorted(arrays),
        })
        return self.result_from_arrays(arrays)

    def _run_tool(self, image_path: str, out_dir: str):  # pragma: no cover
        """Run the external tool once; return (rc, stdout, stderr, npz path)."""
        completed = subprocess.run(
            self.command(image_path, out_dir),
            cwd=os.path.join(self.home, "TruFor_train_test"),
            capture_output=True, text=True, timeout=self.timeout_s,
        )
        outputs = sorted(glob.glob(os.path.join(out_dir, "**", "*.npz"), recursive=True))
        return completed.returncode, completed.stdout, completed.stderr, (
            outputs[0] if outputs else "")

    def timing_summary(self) -> dict:
        """Cold vs warm, for the batch-vs-resident decision the review asked for."""
        if not self.timings:
            return {"runs": 0}
        cold = [t["elapsed_s"] for t in self.timings if t["cold_start"] and not t["cached"]]
        warm = [t["elapsed_s"] for t in self.timings if not t["cold_start"] and not t["cached"]]
        cached = [t for t in self.timings if t["cached"]]
        return {
            "runs": self._runs,
            "cold_start_s": cold[0] if cold else None,
            "warm_mean_s": round(sum(warm) / len(warm), 2) if warm else None,
            "warm_min_s": min(warm) if warm else None,
            "cache_hits": len(cached),
        }

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
    Deterministic TruFor stand-in: same class and the same `_score` path.

    Only `_run_tool` is replaced — it writes the npz the real tool would write
    and returns its path — so the cache key, the persistence, the metadata and
    the timing all run through production code.  A mock that took a shortcut
    around them would not notice when they break.
    """

    weight_source = "mock"
    licence_note = "mock（不涉及上游权重）"
    score_reference = "TruFor 检测头（Mock）"

    def __init__(self, score: Optional[float] = None, shape=(64, 64), **kwargs):
        kwargs.setdefault("weights_path", "/dev/null")
        kwargs.setdefault("results_dir", tempfile.mkdtemp(prefix="trufor_mock_"))
        super().__init__(**kwargs)
        self._fixed_score = score
        self._shape = shape
        self._last = None
        self.tool_calls = 0

    @classmethod
    def _resolve_weights_path(cls) -> str:
        return "/dev/null"

    def unavailable_reason(self) -> str:
        return "" if self.enabled else f"{self.source_name} 已停用"

    def _run_tool(self, image_path: str, out_dir: str):
        self.tool_calls += 1
        height, width = self._shape
        rows = np.linspace(0.0, 1.0, height).reshape(-1, 1)
        arrays = {
            "score": np.array(0.5 if self._fixed_score is None else self._fixed_score),
            "map": np.repeat(rows, width, axis=1).astype(np.float32),
            "conf": np.full((height, width), 0.8, dtype=np.float32),
            "imgsize": np.array([height, width]),
        }
        path = os.path.join(out_dir, "mock.npz")
        np.savez(path, **arrays)
        return 0, "mock", "", path
