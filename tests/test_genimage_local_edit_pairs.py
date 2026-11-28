"""Checks for the 2026-11-28 GenImage whole/local edit pair batch."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "sft_data/genimage_local_edits_20261128"


def test_manifest_and_pair_files() -> None:
    manifest = json.loads((BATCH / "manifest.json").read_text(encoding="utf-8"))
    pairs = manifest["pairs"]
    assert 10 <= len(pairs) <= 15
    assert manifest["counts"]["pairs"] == len(pairs)
    assert {pair["source_id"] for pair in pairs} == set(
        pair["source_id"] for pair in pairs)
    assert {pair["edit_type"] for pair in pairs} == {"add", "delete", "replace"}
    assert all(pair["whole_image_truth"] == "AI-generated" for pair in pairs)
    assert all(pair["local_truth_original"] == "Not tampered" for pair in pairs)
    assert all(pair["local_truth_edited"] == "Tampered" for pair in pairs)
    assert all(pair["non_edit_pixels_changed_before_encode"] == 0
               for pair in pairs)

    source_manifest = json.loads(
        (ROOT / manifest["source_manifest"]).read_text(encoding="utf-8"))
    source_index = {row["source_id"]: row for row in source_manifest["inputs"]}
    for pair in pairs:
        meta = source_index[pair["source_id"]]
        assert meta["new_split"] == "dev"
        assert meta["no_known_use"] is True
        for key in ("original_path", "edited_path", "mask_path"):
            assert (BATCH / pair[key]).is_file()
        with Image.open(BATCH / pair["original_path"]) as original:
            assert original.size == (pair["width"], pair["height"])
            assert original.quantization == Image.open(
                BATCH / pair["edited_path"]).quantization
        with Image.open(BATCH / pair["edited_path"]) as edited:
            assert edited.size == (pair["width"], pair["height"])
        mask = np.array(Image.open(BATCH / pair["mask_path"]))
        assert set(np.unique(mask)).issubset({0, 255})
        ys, xs = np.where(mask > 0)
        assert [int(xs.min()), int(ys.min()), int(xs.max()) + 1,
                int(ys.max()) + 1] == pair["mask_bbox_xyxy"]
        assert int((mask > 0).sum()) == pair["mask_pixels"]


def test_review_page_references_existing_images() -> None:
    html = (BATCH / "review/index.html").read_text(encoding="utf-8")
    sources = re.findall(r'src="([^"]+)"', html)
    assert len(sources) == 30
    for rel in sources:
        assert (BATCH / "review" / rel).resolve().is_file()
