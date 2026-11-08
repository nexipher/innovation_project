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
    """Local manipulation score + diagnostic maps, run as a subprocess."""

    source_name = "trufor_expert"
    evidence_name = "manipulation_anomaly_score"
    # Not "整图判别分数": this head answers local manipulation, and the
    # Chinese label in the token is what the model reads first.
    phenomenon_label = "局部篡改检测分数"
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
        #: title/colour-bar/content-box metadata for the last rendered maps
        self.last_artifact_layout: Dict[str, dict] = {}
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
            returncode, stdout, stderr, npz_path = self._run_tool_single(image_path, out_dir)
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

    # ------------------------------------------------------------------
    # Batch: one process for many images
    # ------------------------------------------------------------------

    def analyze_many(self, images: Dict[str, np.ndarray],
                     progress: bool = False) -> Dict[str, object]:
        """
        Score several images in **one** external process.

        Measured on this box: a single image costs ~12 s, of which 12.2 s is
        the cold start and 12.2 s the warm call — i.e. the interpreter, the
        imports and the checkpoint, not the inference.  Their `test.py` takes a
        directory, so the batch path pays that once: the fitting group's 464
        images drop from ~1.6 h to minutes.
        """
        from PIL import Image

        results: Dict[str, object] = {}
        pending: Dict[str, np.ndarray] = {}
        keys: Dict[str, str] = {}
        for image_id, image in images.items():
            key = self.cache_key(image)
            keys[image_id] = key
            cached = self.cached_result(key)
            if cached is not None:
                self._last = cached
                self.timings.append({"key": key, "cold_start": False,
                                     "elapsed_s": 0.0, "cached": True})
                results[image_id] = self.result_from_arrays(cached)
            else:
                pending[image_id] = image

        if not pending:
            return results

        started = time.perf_counter()
        cold_start = self._runs == 0
        with tempfile.TemporaryDirectory(prefix="trufor_batch_") as work:
            in_dir = os.path.join(work, "in")
            out_dir = os.path.join(work, "out")
            os.makedirs(in_dir, exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)
            for image_id, image in pending.items():
                path = os.path.join(in_dir, f"{keys[image_id]}.png")
                Image.fromarray(np.ascontiguousarray(image[:, :, ::-1])).save(path)
            returncode, stdout, stderr = self._run_tool_directory(in_dir, out_dir)
            # `test.py` derives the output name from the input path
            # (`root = input.split('*')[0]`, then `path.replace(root, '')`), and
            # for a directory input that can resolve to an absolute path — the
            # npz then lands beside the input rather than under `-out`.  Both
            # are ours, so search the whole work directory instead of guessing.
            produced = {}
            for path in glob.glob(os.path.join(work, "**", "*.npz"), recursive=True):
                produced[os.path.splitext(os.path.basename(path))[0]
                         .replace(".png", "")] = path
            missing = [image_id for image_id in pending if keys[image_id] not in produced]
            if missing:
                raise RuntimeError(
                    f"{self.source_name}：批量运行缺少 {len(missing)} 张的结果"
                    f"（退出码 {returncode}）\nstdout 尾部：{stdout[-400:]}\n"
                    f"stderr 尾部：{stderr[-400:]}")

            for image_id, image in pending.items():
                key = keys[image_id]
                arrays = dict(np.load(produced[key], allow_pickle=True))
                self.persist(key, arrays, {
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "image_sha256": hashlib.sha256(
                        np.ascontiguousarray(image).tobytes()).hexdigest(),
                    "weights": self.provenance().as_dict(),
                    "experiment": self.experiment,
                    "recipe_version": RECIPE_VERSION,
                    "preprocess": "原图 PNG 落盘 → TruFor test.py 批量目录前向",
                    "high_response_threshold": HIGH_RESPONSE_THRESHOLD,
                    "cold_start": cold_start and image_id == next(iter(pending)),
                    "batch_size": len(pending),
                    "subprocess": {"returncode": returncode,
                                   "stdout_tail": stdout[-400:],
                                   "stderr_tail": stderr[-400:]},
                    "stats": self.summarise(arrays),
                    "npz_keys": sorted(arrays),
                })
                self._last = arrays
                results[image_id] = self.result_from_arrays(arrays)

        elapsed = time.perf_counter() - started
        self._runs += len(pending)
        per_image = round(elapsed / len(pending), 2)
        self.timings.append({
            "key": f"batch-{len(pending)}", "cold_start": cold_start,
            "elapsed_s": round(elapsed, 2), "per_image_s": per_image,
            "cached": False, "batch_size": len(pending),
        })
        if progress:
            print(f"    批量 {len(pending)} 张，共 {elapsed:.1f}s（{per_image}s/张）")
        return results

    def _run_tool_directory(self, in_dir: str, out_dir: str):  # pragma: no cover
        """One invocation for a whole directory; return (rc, stdout, stderr)."""
        completed = subprocess.run(
            self.command(in_dir, out_dir),
            cwd=os.path.join(self.home, "TruFor_train_test"),
            capture_output=True, text=True, timeout=self.timeout_s,
        )
        return completed.returncode, completed.stdout, completed.stderr

    def _run_tool_single(self, image_path: str, out_dir: str):  # pragma: no cover
        """Run the external tool for one image; return (rc, stdout, stderr, path)."""
        completed = subprocess.run(
            self.command(image_path, out_dir),
            cwd=os.path.join(self.home, "TruFor_train_test"),
            capture_output=True, text=True, timeout=self.timeout_s,
        )
        outputs = sorted(glob.glob(os.path.join(out_dir, "**", "*.npz"), recursive=True))
        return completed.returncode, completed.stdout, completed.stderr, (
            outputs[0] if outputs else "")

    def timing_summary(self) -> dict:
        """
        Cold vs warm, **per image**, for the batch-vs-resident decision.

        A batch call's `elapsed_s` is the whole batch's wall time; reporting it
        next to per-image numbers would make a 0.3 s/image pipeline look like a
        20 s one (and vice versa).  Batch entries therefore report
        `per_image_s`, and the mode is named.
        """
        if not self.timings:
            return {"runs": 0, "mode": "none"}
        batches = [t for t in self.timings if t.get("batch_size")]
        singles = [t for t in self.timings if not t.get("batch_size") and not t["cached"]]
        cached = [t for t in self.timings if t["cached"]]
        warm = [t for t in singles if not t["cold_start"]]

        if batches:
            per_image = [t["per_image_s"] for t in batches]
            images = sum(t["batch_size"] for t in batches)
            return {
                "runs": self._runs, "mode": "batch",
                "images_in_batches": images,
                "batches": len(batches),
                "cold_start_s": (batches[0]["elapsed_s"] if batches[0]["cold_start"]
                                 else None),
                "per_image_mean_s": round(sum(per_image) / len(per_image), 3),
                "per_image_min_s": min(per_image),
                "total_s": round(sum(t["elapsed_s"] for t in batches), 2),
                "cache_hits": len(cached),
            }
        return {
            "runs": self._runs, "mode": "per_image",
            "cold_start_s": next((t["elapsed_s"] for t in singles if t["cold_start"]), None),
            "warm_mean_s": round(sum(t["elapsed_s"] for t in warm) / len(warm), 2) if warm else None,
            "warm_min_s": min(t["elapsed_s"] for t in warm) if warm else None,
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
            peak_y, peak_x = np.unravel_index(int(np.argmax(anomaly)), anomaly.shape)
            height, width = anomaly.shape[:2]
            stats.update({
                "anomaly_peak": float(anomaly.max()),
                "anomaly_mean": float(anomaly.mean()),
                # The peak in the same normalized system as the model's bboxes.
                # Review uses it to flag a predicted box that does not contain
                # the response the report describes; it is not shown to the
                # model as a location to copy.
                "anomaly_peak_normalized_1000": [
                    int(round(peak_y / max(1, height - 1) * 1000)),
                    int(round(peak_x / max(1, width - 1) * 1000)),
                ],
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
        extra = {"threshold": HIGH_RESPONSE_THRESHOLD,
                 "high_response_threshold": HIGH_RESPONSE_THRESHOLD,
                 "has_noiseprint": "np++" in arrays,
                 "stats": stats}
        for key, label in (("map", "map_size"), ("conf", "confidence_map_size")):
            array = arrays.get(key)
            if array is not None:
                extra[label] = [int(array.shape[0]), int(array.shape[1])]
        preprocess = "原图直接前向（TruFor 自行缩放），输出 map/conf/score"
        if size is not None:
            height, width = (int(v) for v in np.asarray(size).reshape(-1)[:2])
            # Audit-only.  The concrete pixel size is deliberately not placed
            # in the model-visible prose: models copied it into the normalized
            # bbox field ([0,0,341,256]).
            extra["image_size"] = [height, width]
        return self._result(
            score, preprocess=preprocess,
            extra=extra,
        )

    @staticmethod
    def _compose_artifact(content: np.ndarray, title: str,
                          high_label: str, low_label: str, *,
                          colour_bar: bool = True,
                          footer: Optional[str] = None):
        """
        Put title and a fixed JET colour bar OUTSIDE the image content.

        The returned content box is ``[x0, y0, x1, y1]`` with exclusive
        ``x1``/``y1``.  Review overlays must map original-image coordinates
        into this box only; the title and legend are not image pixels.

        The right margin is the same whether or not a colour bar is drawn, so
        the original and the two maps can share one canvas size as well as one
        content size.
        """
        import cv2

        height, width = content.shape[:2]
        top, right, bottom = 38, 205, 62
        canvas = np.full((top + height + bottom, width + right, 3), 248,
                         dtype=np.uint8)
        canvas[top:top + height, :width] = content
        cv2.rectangle(canvas, (0, top), (width - 1, top + height - 1),
                      (70, 70, 70), 1)
        # Keep the title inside the canvas even for small source images.
        title_scale = 0.62
        (title_w, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX,
                                         title_scale, 1)
        max_title_w = canvas.shape[1] - 20
        if title_w > max_title_w:
            title_scale *= max_title_w / float(title_w)
        cv2.putText(canvas, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    title_scale, (25, 25, 25), 1, cv2.LINE_AA)

        bar_x = width + 28
        bar_y = top + 20
        bar_w, bar_h = 28, max(40, height - 40)
        if colour_bar:
            for row in range(bar_h):
                # JET: index 0 is blue (low), 255 is red (high).  Rows run
                # top=high to bottom=low.
                frac = 1.0 - row / max(1, bar_h - 1)
                colour = cv2.applyColorMap(
                    np.array([[int(round(255 * frac))]], dtype=np.uint8),
                    cv2.COLORMAP_JET)[0, 0]
                canvas[bar_y + row, bar_x:bar_x + bar_w] = colour
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h),
                          (70, 70, 70), 1)
            cv2.putText(canvas, high_label, (bar_x + bar_w + 10, bar_y + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 25, 25), 1, cv2.LINE_AA)
            cv2.putText(canvas, low_label, (bar_x + bar_w + 10, bar_y + bar_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (25, 25, 25), 1, cv2.LINE_AA)
        else:
            # No colour bar on the original panel: the same right margin keeps
            # the three canvases the same size while the labels say plainly
            # that this panel is the reference image, not a response map.
            cv2.putText(canvas, high_label, (bar_x, top + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (25, 25, 25), 1, cv2.LINE_AA)
            cv2.putText(canvas, low_label, (bar_x, top + 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1, cv2.LINE_AA)
        footer_text = footer or "title and colour bar are outside the content area"
        footer_scale = 0.43
        (footer_w, _), _ = cv2.getTextSize(footer_text, cv2.FONT_HERSHEY_SIMPLEX,
                                          footer_scale, 1)
        max_footer_w = canvas.shape[1] - 20
        if footer_w > max_footer_w:
            footer_scale *= max_footer_w / float(footer_w)
        cv2.putText(canvas, footer_text, (10, top + height + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, footer_scale, (90, 90, 90), 1,
                    cv2.LINE_AA)
        return canvas, [0, top, width, top + height], [canvas.shape[1], canvas.shape[0]]

    def render_region_artifacts(self, img_bgr: np.ndarray,
                                region_pixels: List[int],
                                region_normalized: Optional[List[int]] = None):
        """
        Render one same-content trio from the *last* measurement.

        Original crop, anomaly crop and confidence crop all show the same
        original-image region, at the same pixel size, with a title and legend
        outside the content.  Maps are resized to the original image frame
        before cropping if TruFor returned a different map size.  No model
        call is made here: this is a view of `self._last`.

        Returns ``(artifacts, layout)``.  ``layout`` records the region in
        both coordinate systems and the map→source scale, so a reviewer can
        audit the transform instead of inferring it from the canvas.
        """
        if not self._last:
            return {}, {}
        import cv2

        source_h, source_w = img_bgr.shape[:2]
        try:
            ymin, xmin, ymax, xmax = [int(round(float(v)))
                                      for v in region_pixels]
        except (TypeError, ValueError):
            ymin, xmin, ymax, xmax = 0, 0, source_w, source_h
        ymin = max(0, min(source_h - 1, ymin))
        xmin = max(0, min(source_w - 1, xmin))
        ymax = max(ymin + 1, min(source_h, ymax))
        xmax = max(xmin + 1, min(source_w, xmax))
        if ymax <= ymin or xmax <= xmin:
            ymin, xmin, ymax, xmax = 0, 0, source_h, source_w
        full = (ymin, xmin, ymax, xmax) == (0, 0, source_h, source_w)
        if region_normalized is None:
            # Bbox edges divide by the image size; only a pixel *centre*
            # (the anomaly peak) uses size-1.  Using size-1 here made a full
            # 333x500 frame read as [0,0,1003,1002].
            region_normalized = [
                int(round(ymin / max(1, source_h) * 1000)),
                int(round(xmin / max(1, source_w) * 1000)),
                int(round(ymax / max(1, source_h) * 1000)),
                int(round(xmax / max(1, source_w) * 1000)),
            ]
        else:
            region_normalized = [
                max(0, min(1000, int(v))) for v in region_normalized]

        map_array = self._last.get("map")
        conf_array = self._last.get("conf")
        map_size = ([int(map_array.shape[0]), int(map_array.shape[1])]
                    if map_array is not None else None)

        def _resize_to_source(array):
            if array is None:
                return None
            array = np.asarray(array, dtype=np.float64)
            if array.shape[:2] == (source_h, source_w):
                return array
            return cv2.resize(array, (source_w, source_h),
                              interpolation=cv2.INTER_LINEAR)

        map_array = _resize_to_source(map_array)
        conf_array = _resize_to_source(conf_array)
        original_crop = img_bgr[ymin:ymax, xmin:xmax].copy()
        anomaly_crop = (map_array[ymin:ymax, xmin:xmax]
                        if map_array is not None else None)
        conf_crop = (conf_array[ymin:ymax, xmin:xmax]
                     if conf_array is not None else None)
        crop_h, crop_w = original_crop.shape[:2]
        if full:
            area_label = "full original-image frame"
            title_suffix = "full frame"
            footer = ("full original-image frame; all three panels share the "
                      "same content size")
        else:
            # Pixel bounds stay in the audit layout, not in the picture text:
            # a footer with [w,h] is another value the model can copy into a
            # normalized bbox.  The panels already show the same crop.
            area_label = "same original-image crop"
            title_suffix = "same region"
            footer = ("same original-image crop; all three panels share the "
                      "same content size")

        artifacts: Dict[str, np.ndarray] = {}
        layout: Dict[str, dict] = {}
        panels = (
            ("original_crop", original_crop,
             f"Original image — {title_suffix}", "ORIGINAL", "SAME CROP",
             False),
        )
        if anomaly_crop is not None:
            panels += (("anomaly_map",
                        cv2.applyColorMap(
                            np.clip(anomaly_crop * 255.0, 0, 255).astype(np.uint8),
                            cv2.COLORMAP_JET),
                        f"TruFor anomaly map — {title_suffix}",
                        "HIGH response (red)", "LOW response (blue)", True),)
        if conf_crop is not None:
            panels += (("confidence_map",
                        cv2.applyColorMap(
                            np.clip(conf_crop * 255.0, 0, 255).astype(np.uint8),
                            cv2.COLORMAP_JET),
                        f"TruFor confidence map — {title_suffix}",
                        "HIGH reliability (red)", "LOW reliability (blue)", True),)
        for name, content, title, high_label, low_label, colour_bar in panels:
            canvas, content_box, canvas_size = self._compose_artifact(
                content, title, high_label, low_label,
                colour_bar=colour_bar, footer=footer)
            artifacts[name] = canvas
            layout[name] = {
                "title": title,
                "content_box": content_box,
                "canvas_size": canvas_size,
                "low_colour": "blue" if colour_bar else None,
                "high_colour": "red" if colour_bar else None,
                "low_label": low_label,
                "high_label": high_label,
                "content_area": area_label,
                "region_pixels": [ymin, xmin, ymax, xmax],
                "region_normalized_1000": region_normalized,
                "source_size": [source_h, source_w],
                "map_size": map_size,
                "crop_size": [crop_h, crop_w],
                "scale_map_to_source": (
                    [round(source_w / map_size[1], 6),
                     round(source_h / map_size[0], 6)]
                    if map_size else None),
            }
        return artifacts, layout

    def render_artifacts(self, img_patch: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Paint the maps from the *last* analysis — no second forward pass.

        The two products are diagnostic: anomaly response and confidence.
        Both get a title and a fixed JET colour bar outside the content area,
        and the content box is recorded so review overlays can map original
        coordinates correctly.  Their wording rule is enforced in the token,
        not by the image.
        """
        if not self._last:
            return {}
        import cv2

        artifacts: Dict[str, np.ndarray] = {}
        self.last_artifact_layout = {}
        specs = (
            ("map", "anomaly_map", "TruFor anomaly map (local manipulation response)",
             "HIGH response (red)", "LOW response (blue)"),
            ("conf", "confidence_map", "TruFor confidence map (reliability)",
             "HIGH reliability (red)", "LOW reliability (blue)"),
        )
        for key, name, title, high_label, low_label in specs:
            array = self._last.get(key)
            if array is None:
                continue
            scaled = np.clip(np.asarray(array, dtype=np.float64) * 255.0,
                             0, 255).astype(np.uint8)
            array = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
            canvas, content_box, canvas_size = self._compose_artifact(
                array, title, high_label, low_label)
            artifacts[name] = canvas
            self.last_artifact_layout[name] = {
                "title": title,
                "content_box": content_box,
                "canvas_size": canvas_size,
                "low_colour": "blue",
                "high_colour": "red",
                "low_label": low_label,
                "high_label": high_label,
            }
        noiseprint = self._last.get("np++")
        if noiseprint is not None:
            magnitude = np.abs(np.asarray(noiseprint, dtype=np.float64))
            peak = magnitude.max() or 1.0
            scaled = np.clip(magnitude / peak * 255.0, 0, 255).astype(np.uint8)
            array = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
            canvas, content_box, canvas_size = self._compose_artifact(
                array, "TruFor Noiseprint++ (diagnostic)",
                "HIGH magnitude (red)", "LOW magnitude (blue)")
            artifacts["noiseprint_pp"] = canvas
            self.last_artifact_layout["noiseprint_pp"] = {
                "title": "TruFor Noiseprint++ (diagnostic)",
                "content_box": content_box,
                "canvas_size": canvas_size,
                "low_colour": "blue",
                "high_colour": "red",
                "low_label": "LOW magnitude (blue)",
                "high_label": "HIGH magnitude (red)",
            }
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

    def _arrays(self):
        height, width = self._shape
        rows = np.linspace(0.0, 1.0, height).reshape(-1, 1)
        return {
            "score": np.array(0.5 if self._fixed_score is None else self._fixed_score),
            "map": np.repeat(rows, width, axis=1).astype(np.float32),
            "conf": np.full((height, width), 0.8, dtype=np.float32),
            "imgsize": np.array([height, width]),
        }

    def _run_tool_single(self, image_path: str, out_dir: str):
        self.tool_calls += 1
        path = os.path.join(out_dir, "mock.npz")
        np.savez(path, **self._arrays())
        return 0, "mock", "", path

    def _run_tool_directory(self, in_dir: str, out_dir: str):
        """One batch call, one npz per input file — as the real tool behaves."""
        self.tool_calls += 1
        for name in sorted(os.listdir(in_dir)):
            stem = os.path.splitext(name)[0]
            np.savez(os.path.join(out_dir, f"{stem}.npz"), **self._arrays())
        return 0, "mock-batch", ""
