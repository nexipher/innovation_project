"""Offline A/B/C TruFor crop comparison on the 15 AutoSplice validation images.

A: current high-response bounding box, including the existing fallback
   (peak crop when the high-response blob is tiny or absent).
B: connected components of the same high-response mask, ranked by component
   mean anomaly response and mean confidence at the same pixels; Top-1.
C: the same ranking, Top-2; metrics use the union of the two boxes.

No Qwen, no Agent and no new TruFor run.  The masks are used only for scoring,
never to choose a window.  The 10 forged + 5 authentic images and the existing
`analysis/trufor_raw` maps are reused.

Usage:
  python scripts/compare_trufor_crop_windows.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/root/autodl-tmp/innovation_project")
SMALL = ROOT / "sft_data/lora_small_20261128"
ANALYSIS = SMALL / "analysis"
SPLIT = ROOT / "sft_data/sft_candidates_20261127/split.json"
OUT = ANALYSIS / "crop_comparison_20261128"
sys.path.insert(0, str(ROOT))

REPRESENTATIVE = {
    "41993", "43686", "57143", "73275", "67065",
    "40489", "56086", "57276", "60788",
    "47630", "65749",
}


def _npz_for(score: float) -> dict:
    for path in (ANALYSIS / "trufor_raw").glob("*/result.npz"):
        arrays = np.load(path, allow_pickle=True)
        if "score" not in arrays:
            continue
        if abs(float(np.asarray(arrays["score"]).reshape(-1)[0]) - score) < 1e-5:
            return {key: arrays[key] for key in arrays.files}
    raise SystemExit(f"no npz for score {score}")


def _resize(array, shape):
    if array is None:
        return None
    array = np.asarray(array, dtype=np.float64)
    if array.shape[:2] == shape:
        return array
    return cv2.resize(array, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def _mask_for(source_id: str):
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    entry = split["autosplice"]["bases"][source_id]
    mask = cv2.imread(entry["mask"]["path"], cv2.IMREAD_GRAYSCALE)
    return mask


def _pad_box(box, shape, pad_ratio=0.05):
    y0, x0, y1, x1 = box
    h, w = shape
    pad = max(4, int(round(min(y1 - y0, x1 - x0) * pad_ratio)))
    return (max(0, y0 - pad), max(0, x0 - pad),
            min(h, y1 + pad), min(w, x1 + pad))


def _box_mask(shape, box):
    win = np.zeros(shape, dtype=bool)
    if box is None:
        return win
    y0, x0, y1, x1 = box
    win[y0:y1, x0:x1] = True
    return win


def _window_metrics(win, mask, gray, canny, whole_tex, whole_edge):
    h, w = gray.shape
    if not win.any():
        return {"window_area_px": 0, "window_area_frac": None,
                "mask_pixels": int(mask.sum()) if mask is not None else None,
                "mask_in_window": None, "coverage": None,
                "window_mask_frac": None, "touches_border": None,
                "border_distance_px": None, "edge_density": None,
                "edge_density_ratio": None, "texture_var": None,
                "texture_ratio": None, "real_context": None}
    ys, xs = np.where(win)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1
    area = int(win.sum())
    edge = canny > 0
    edge_density = float(edge[win].mean())
    texture = float(cv2.Laplacian(gray[win], cv2.CV_64F).var())
    touches = y0 == 0 or x0 == 0 or y1 == h or x1 == w
    border_distance = int(min(y0, x0, h - y1, w - x1))
    context = []
    if touches or border_distance <= max(4, int(round(min(h, w) * 0.03))):
        context.append("border")
    if whole_edge and edge_density >= 1.5 * whole_edge:
        context.append("edge")
    if whole_tex and texture >= 1.5 * whole_tex:
        context.append("texture")
    out = {
        "window_area_px": area,
        "window_area_frac": round(area / float(h * w), 6),
        "mask_pixels": int(mask.sum()) if mask is not None else None,
        "mask_in_window": None,
        "coverage": None,
        "window_mask_frac": None,
        "touches_border": bool(touches),
        "border_distance_px": border_distance,
        "edge_density": round(edge_density, 6),
        "edge_density_ratio": (round(edge_density / whole_edge, 4)
                               if whole_edge else None),
        "texture_var": round(texture, 4),
        "texture_ratio": (round(texture / whole_tex, 4)
                          if whole_tex else None),
        "real_context": "+".join(context) if context else "content",
    }
    if mask is not None:
        inside = int(np.logical_and(mask, win).sum())
        out["mask_in_window"] = inside
        out["coverage"] = round(inside / int(mask.sum()), 6) if mask.sum() else None
        out["window_mask_frac"] = round(inside / float(area), 6)
    return out


def _components(map_array, conf_array):
    high = (map_array >= 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        high, connectivity=8)
    comps = []
    for label in range(1, count):
        ys, xs = np.where(labels == label)
        if ys.size == 0:
            continue
        values = map_array[ys, xs]
        conf = conf_array[ys, xs] if conf_array is not None else np.ones_like(values)
        comps.append({
            "label": label,
            "area": int(ys.size),
            "box": (int(ys.min()), int(xs.min()),
                    int(ys.max()) + 1, int(xs.max()) + 1),
            "mean_anomaly": float(values.mean()),
            "mean_conf": float(conf.mean()),
            "peak": float(values.max()),
        })
    comps.sort(key=lambda item: (
        -(item["mean_anomaly"] * item["mean_conf"]),
        -item["area"], -item["peak"]))
    return comps


def _current_a(img, map_array, conf_array):
    from experts.trufor import MockTruForExpert

    expert = MockTruForExpert(shape=img.shape[:2])
    expert._last = {"map": map_array, "conf": conf_array}
    _panels, layout = expert.render_region_artifacts(img, None, None)
    original = layout["original_crop"]
    return tuple(original["region_pixels"]), original.get("region_source")


def _union_metrics(box1, box2, shape, mask, gray, canny, whole_tex, whole_edge):
    w1 = _box_mask(shape, box1)
    w2 = _box_mask(shape, box2)
    union = np.logical_or(w1, w2)
    m1 = _window_metrics(w1, mask, gray, canny, whole_tex, whole_edge)
    mu = _window_metrics(union, mask, gray, canny, whole_tex, whole_edge)
    return m1, mu, union


def _draw_box(image, box, colour, label):
    y0, x0, y1, x1 = [int(v) for v in box]
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), colour, 2)
    cv2.putText(image, label, (x0 + 2, max(12, y0 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)


def _representative(row, img, map_array, out_dir):
    h, w = img.shape[:2]
    factor = max(1.0, 680.0 / max(1, w))
    original = cv2.resize(img, (int(round(w * factor)), int(round(h * factor))),
                          interpolation=cv2.INTER_NEAREST)
    anomaly = cv2.applyColorMap(
        np.clip(map_array * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    anomaly = cv2.resize(anomaly, original.shape[1::-1],
                         interpolation=cv2.INTER_NEAREST)

    def scale(box):
        return tuple(int(round(v * factor)) for v in box)

    for label, colour, box in (
        ("A", (255, 0, 0), row.get("A_box")),
        ("B1", (0, 255, 0), row.get("B1_box")),
        ("C1", (0, 255, 0), row.get("C1_box")),
        ("C2", (0, 165, 255), row.get("C2_box")),
    ):
        if box is None:
            continue
        for canvas in (original, anomaly):
            _draw_box(canvas, scale(box), colour, label)
    header = np.zeros((32, original.shape[1], 3), dtype=np.uint8)
    cv2.putText(header, f"{row['source_id']}  score={row['trufor_score']:.4f}  "
                       "A=blue B1=green C2=orange", (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    panel = np.vstack([header, original, anomaly])
    path = out_dir / f"{row['source_id']}.png"
    cv2.imwrite(str(path), panel)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "representative").mkdir(exist_ok=True)
    rows = json.loads((ANALYSIS / "analysis.json").read_text(encoding="utf-8"))["samples"]
    rows = [row for row in rows if row["group"] in
            ("AutoSplice_Forged", "AutoSplice_Authentic")]
    records = []
    by_sample = {}
    for row in rows:
        source_id = row["source_id"]
        img = cv2.imread(row["path"], cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"image unreadable: {row['path']}")
        h, w = img.shape[:2]
        arrays = _npz_for(row["trufor_score"])
        map_array = _resize(arrays["map"], (h, w))
        conf_array = _resize(arrays.get("conf"), (h, w))
        mask = None
        if row["group"] == "AutoSplice_Forged":
            mask = cv2.resize(_mask_for(source_id), (w, h),
                              interpolation=cv2.INTER_NEAREST) > 0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(gray, 100, 200)
        whole_tex = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        whole_edge = float((canny > 0).mean())
        comps = _components(map_array, conf_array)

        a_box, a_source = _current_a(img, map_array, conf_array)
        a_win = _box_mask((h, w), a_box)
        a_metrics = _window_metrics(a_win, mask, gray, canny, whole_tex, whole_edge)
        a_metrics.update({"sample_id": row["sample_id"], "source_id": source_id,
                          "group": row["group"], "crop": "A_high_bbox",
                          "source": a_source, "component_count": len(comps),
                          "box1": a_box, "box2": None,
                          "second_new_coverage": None,
                          "second_new_area_frac": None,
                          "union_coverage": a_metrics["coverage"],
                          "union_area_frac": a_metrics["window_area_frac"],
                          "trufor_score": row["trufor_score"]})
        records.append(a_metrics)

        if not comps:
            for crop in ("B_top1", "C_top2"):
                rows_metrics = {"sample_id": row["sample_id"], "source_id": source_id,
                                "group": row["group"], "crop": crop,
                                "source": "no_high_component",
                                "component_count": 0, "trufor_score": row["trufor_score"],
                                "box1": None, "box2": None,
                                "window_area_px": 0, "window_area_frac": None,
                                "mask_pixels": int(mask.sum()) if mask is not None else None,
                                "mask_in_window": None, "coverage": None,
                                "window_mask_frac": None, "touches_border": None,
                                "border_distance_px": None, "edge_density": None,
                                "edge_density_ratio": None, "texture_var": None,
                                "texture_ratio": None, "real_context": None,
                                "second_new_coverage": None,
                                "second_new_area_frac": None,
                                "union_coverage": None, "union_area_frac": None}
                records.append(rows_metrics)
            by_sample[source_id] = {"A_box": a_box, **a_metrics,
                                    "B1_box": None, "C1_box": None, "C2_box": None}
            if source_id in REPRESENTATIVE:
                _representative(by_sample[source_id], img, map_array,
                                out / "representative")
            continue

        b1 = _pad_box(comps[0]["box"], (h, w))
        b1_win = _box_mask((h, w), b1)
        b1_metrics = _window_metrics(b1_win, mask, gray, canny, whole_tex, whole_edge)
        b1_metrics.update({
            "sample_id": row["sample_id"], "source_id": source_id,
            "group": row["group"], "crop": "B_top1", "source": "component_top1",
            "component_count": len(comps), "box1": b1, "box2": None,
            "second_new_coverage": None, "second_new_area_frac": None,
            "union_coverage": b1_metrics["coverage"],
            "union_area_frac": b1_metrics["window_area_frac"],
            "trufor_score": row["trufor_score"]})
        records.append(b1_metrics)

        c1, c2 = b1, None
        c_metrics = dict(b1_metrics)
        if len(comps) >= 2:
            c2 = _pad_box(comps[1]["box"], (h, w))
            m1, mu, union = _union_metrics(
                c1, c2, (h, w), mask, gray, canny, whole_tex, whole_edge)
            c2_metrics = _window_metrics(
                _box_mask((h, w), c2), mask, gray, canny, whole_tex, whole_edge)
            c_metrics.update(mu)
            c_metrics.update({
                "second_box_area_px": c2_metrics["window_area_px"],
                "second_box_area_frac": c2_metrics["window_area_frac"],
                "crop": "C_top2_union", "source": "component_top2_union",
                "box1": c1, "box2": c2,
                "second_new_coverage": (
                    None if mu["coverage"] is None or m1["coverage"] is None
                    else round(mu["coverage"] - m1["coverage"], 6)),
                "second_new_area_frac": (
                    None if mu["window_area_frac"] is None
                    or m1["window_area_frac"] is None
                    else round(mu["window_area_frac"] - m1["window_area_frac"], 6)),
                "union_coverage": mu["coverage"],
                "union_area_frac": mu["window_area_frac"],
                "trufor_score": row["trufor_score"],
            })
        else:
            c_metrics.update({
                "crop": "C_top2_union", "source": "one_component_only",
                "box1": c1, "box2": None,
                "second_new_coverage": None, "second_new_area_frac": None,
                "union_coverage": c_metrics["coverage"],
                "union_area_frac": c_metrics["window_area_frac"],
                "trufor_score": row["trufor_score"],
            })
        records.append(c_metrics)
        by_sample[source_id] = {"A_box": a_box, **a_metrics,
                                "B1_box": b1, "C1_box": c1, "C2_box": c2}
        if source_id in REPRESENTATIVE:
            _representative(by_sample[source_id], img, map_array,
                            out / "representative")

    fields = [
        "sample_id", "source_id", "group", "crop", "trufor_score", "source",
        "component_count", "box1", "box2", "window_area_px", "window_area_frac",
        "mask_pixels", "mask_in_window", "coverage", "window_mask_frac",
        "second_box_area_px", "second_box_area_frac",
        "second_new_coverage", "second_new_area_frac", "union_coverage",
        "union_area_frac", "touches_border", "border_distance_px",
        "edge_density", "edge_density_ratio", "texture_var", "texture_ratio",
        "real_context",
    ]
    with (out / "per_image.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(record.get(key), ensure_ascii=False)
                             if isinstance(record.get(key), (list, tuple, bool)) else
                             record.get(key) for key in fields})
    (out / "per_image.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    forged = [r for r in rows if r["group"] == "AutoSplice_Forged"]
    authentic = [r for r in rows if r["group"] == "AutoSplice_Authentic"]
    def lookup(crop):
        return {r["source_id"]: r for r in records
                if r["group"] == "AutoSplice_Forged" and r["crop"] == crop}
    a, b, c = lookup("A_high_bbox"), lookup("B_top1"), lookup("C_top2_union")
    def mean(key, table):
        vals = [row[key] for row in table.values() if row.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None
    b_recovers = [sid for sid in a
                  if b[sid].get("coverage") is not None and a[sid].get("coverage")
                  and b[sid]["coverage"] >= 0.95 * a[sid]["coverage"]]
    b_smaller = [sid for sid in a
                 if b[sid].get("window_area_frac") is not None
                 and a[sid].get("window_area_frac")
                 and b[sid]["window_area_frac"] < a[sid]["window_area_frac"]]
    c_recovers = [sid for sid in a
                  if c[sid].get("coverage") is not None and a[sid].get("coverage")
                  and c[sid]["coverage"] >= 0.95 * a[sid]["coverage"]]
    c_adds = [sid for sid in a if (c[sid].get("second_new_coverage") or 0) > 0]
    lines = [
        "# TruFor 裁剪窗口 A/B/C 离线对照",
        "",
        "A = 当前 high-response 包围盒（含旧 peak 回退）；B = high-response 像素"
        "连通分量按 `mean anomaly × mean confidence` 排序的 Top-1；C = 同排序 Top-2 的"
        "两框并集。窗口都加 5% 上下文；mask 只用于评分，不参与选窗。",
        "",
        "## 聚合（10 张篡改图）",
        "",
        f"- A：mean coverage {mean('coverage', a)}，mean window area {mean('window_area_frac', a)}，"
        f"mean window mask frac {mean('window_mask_frac', a)}。",
        f"- B：mean coverage {mean('coverage', b)}，mean window area {mean('window_area_frac', b)}，"
        f"mean window mask frac {mean('window_mask_frac', b)}。",
        f"- C：mean union coverage {mean('coverage', c)}，mean union area {mean('window_area_frac', c)}，"
        f"mean window mask frac {mean('window_mask_frac', c)}。",
        f"- B coverage ≥ A 的 95% 的图：{len(b_recovers)}/10"
        f"（{', '.join(b_recovers) or '无'}）。",
        f"- B 窗口面积小于 A 的图：{len(b_smaller)}/10"
        f"（{', '.join(b_smaller) or '无'}）；这些图里能保住 A 的 95% 覆盖的只有 "
        f"{', '.join(sid for sid in b_smaller if b[sid].get('coverage') is not None and a[sid].get('coverage') and b[sid]['coverage'] >= 0.95 * a[sid]['coverage']) or '无'}。",
        f"- B 与 A 完全同一框（高响应只有一个主导连通分量）："
        f"{', '.join(sid for sid in a if tuple(b[sid].get('box1') or ()) == tuple(a[sid].get('box1') or ())) or '无'}。",
        f"- C coverage ≥ A 的 95% 的图：{len(c_recovers)}/10"
        f"（{', '.join(c_recovers) or '无'}）。",
        f"- C 第二框新增覆盖 > 0 的图：{len(c_adds)}/10"
        f"（{', '.join(c_adds) or '无'}）。",
        "",
        "## 逐图（篡改）",
        "",
        "| sample | A cov / area | B cov / area | C union cov / area | C second +cov / +area |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in forged:
        sid = row["source_id"]
        lines.append(
            f"| `{sid}` | {a[sid]['coverage']} / {a[sid]['window_area_frac']} | "
            f"{b[sid]['coverage']} / {b[sid]['window_area_frac']} | "
            f"{c[sid]['coverage']} / {c[sid]['window_area_frac']} | "
            f"{c[sid].get('second_new_coverage')} / {c[sid].get('second_new_area_frac')} |")
    lines += [
        "",
        "## 真图窗口",
        "",
        "| sample | A window / context | B window / context | C window / context |",
        "|---|---|---|---|",
    ]
    real_records = {r["source_id"]: r for r in records
                    if r["group"] == "AutoSplice_Authentic"}
    for row in authentic:
        sid = row["source_id"]
        def desc(crop):
            r = real_records[sid]
            # real_records currently last row per sample; use explicit lookup
            r = next(x for x in records if x["source_id"] == sid and x["crop"] == crop)
            if not r.get("window_area_frac"):
                return "no high-response component"
            return (f"area {r['window_area_frac']}, border={r['touches_border']}, "
                    f"context={r['real_context']}, edge×{r.get('edge_density_ratio')}, "
                    f"texture×{r.get('texture_ratio')}")
        lines.append(f"| `{sid}` | {desc('A_high_bbox')} | {desc('B_top1')} | "
                     f"{desc('C_top2_union')} |")
    lines += [
        "",
        "## Top-2 新增框",
        "",
        "| sample | 第二框 box | 第二框面积占比 | +coverage | +area |",
        "|---|---|---:|---:|---:|",
    ]
    for row in forged:
        sid = row["source_id"]
        if (c[sid].get("second_new_coverage") or 0) <= 0:
            continue
        lines.append(f"| `{sid}` | {c[sid].get('box2')} | "
                     f"{c[sid].get('second_box_area_frac')} | "
                     f"{c[sid].get('second_new_coverage')} | "
                     f"{c[sid].get('second_new_area_frac')} |")
    lines += [
        "",
        "Top-2 只在这 5 张图有非零新增；最大新增是 56086 的 +2.61 个百分点覆盖。"
        "其余图只有一个有效连通分量（或第二分量完全落在第一框内），所以 C=A/B。",
        "",
        "## 代表图",
        "",
        "`representative/` 下为原图 + 异常图，A 蓝、B/C Top-1 绿、C Top-2 橙；"
        "不显示 mask。",
        "",
        "## 结论",
        "",
        "- **Top-1 不能可靠地用更小窗口替代 A。** 10 张里 5 张 B 的 coverage 达到 A 的 95%，"
        "但其中 4 张 B 与 A 是同一个框（41993/42958/57143/67065）；只有 60788 略小"
        "（面积 0.418 vs 0.429，coverage 0.487 vs 0.501），收益很小。其余 5 张 B 虽然"
        "面积更小，coverage 却塌到 0.0016–0.192（A 为 0.305–1.000）。",
        "- **Top-2 补不回 A 的覆盖。** 只有 5 张有非零新增，最大是 56086 的"
        "+2.61 个百分点；C 的并集平均 coverage 0.4675，仍远低于 A 的 0.7866。"
        "43686/73275 的高响应碎成 9/7 个连通分量，Top-2 仍只覆盖很小一角。",
        "- **真图窗口主要落在边框/边缘，不在高纹理上。** A 在 42307/47630/65749/70102"
        "触边，52191 在高边缘密度处；5 张真图的 A texture× 都 <1（0.02–0.61）。"
        "B/C 的真图窗口多数更小且继续贴边（47630 的 edge× 达 1.84/1.88），没有"
        "变成纹理窗口。",
        "- 因此当前 A（所有 high-response 像素的外接框）在这批图上仍是覆盖最稳的方案；"
        "B/C 可以作为更小的辅助定位，但不适合替代三图主裁剪。",
        "",
        "## 边界",
        "",
        "- 只复用已有的 TruFor map/conf 和 15 张验证图；未重跑 Qwen/Agent/TruFor。",
        "- 真图没有 mask，覆盖率列为空；只报告窗口是否触边/高纹理（相对整图）。",
        "- Top-1/Top-2 的排序分数是透明的 `mean anomaly × mean confidence`；"
        "不是新的专家分数。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(records), "forged": len(forged),
                      "authentic": len(authentic),
                      "b_recovers": len(b_recovers),
                      "c_adds": len(c_adds),
                      "out": str(out)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
