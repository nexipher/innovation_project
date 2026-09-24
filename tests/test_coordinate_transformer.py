"""Unit tests for coordinate transformer."""

from utils.coordinate_transformer import CoordinateTransformer as CT


class TestRelativeToAbsolute:
    def test_center_crop(self):
        """[200, 150, 800, 750] in a 500x750 image."""
        abs_bbox = CT.relative_to_absolute([200, 150, 800, 750], 500, 750)
        assert abs_bbox == [150, 75, 600, 375]

    def test_full_image(self):
        abs_bbox = CT.relative_to_absolute([0, 0, 1000, 1000], 512, 512)
        assert abs_bbox == [0, 0, 512, 512]

    def test_top_left(self):
        abs_bbox = CT.relative_to_absolute([0, 0, 500, 500], 1000, 1000)
        assert abs_bbox == [0, 0, 500, 500]

    def test_rounding(self):
        abs_bbox = CT.relative_to_absolute([100, 100, 900, 900], 300, 300)
        assert abs_bbox == [30, 30, 270, 270]


class TestAbsoluteToRelative:
    def test_round_trip(self):
        """rel → abs → rel should be identity (up to rounding)."""
        original = [200, 150, 800, 750]
        abs_bbox = CT.relative_to_absolute(original, 500, 750)
        back = CT.absolute_to_relative(abs_bbox, 500, 750)
        assert back == original

    def test_clamp_to_range(self):
        back = CT.absolute_to_relative([-10, -10, 2000, 2000], 1000, 1000)
        for v in back:
            assert 0 <= v <= 1000


class TestClipBbox:
    def test_within_bounds(self):
        clipped = CT.clip_bbox([100, 100, 300, 300], 500, 500)
        assert clipped == [100, 100, 300, 300]

    def test_negative_coords(self):
        clipped = CT.clip_bbox([-50, -50, 600, 600], 500, 500)
        assert clipped == [0, 0, 500, 500]

    def test_too_small(self):
        """Tiny bbox should be expanded to minimum size."""
        clipped = CT.clip_bbox([100, 100, 105, 105], 500, 500)
        assert clipped[2] - clipped[0] >= 16
        assert clipped[3] - clipped[1] >= 16

    def test_swapped_coords(self):
        """ymin > ymax should be corrected."""
        clipped = CT.clip_bbox([300, 100, 200, 300], 500, 500)
        assert clipped[0] < clipped[2]


class TestTransform:
    """G1 (§4.8): dual-space transform with explicit bookkeeping."""

    def test_returns_both_spaces(self):
        result = CT.transform([200, 150, 800, 750], 500, 750)
        assert result["region_normalized_1000"] == [200, 150, 800, 750]
        assert result["region_pixels"] == [150, 75, 600, 375]
        assert result["coordinate_space"] == "pixels"
        assert result["clipped"] is False

    def test_roundtrip_no_second_scaling(self):
        """pixels -> normalized -> pixels must be stable across sizes."""
        for width, height in [(500, 750), (1024, 1024), (256, 256), (128, 128), (300, 800)]:
            rel = [120, 80, 880, 620]
            first = CT.transform(rel, width, height)
            back = CT.absolute_to_relative(first["region_pixels"], width, height)
            second = CT.transform(back, width, height)
            # A second conversion of the recovered normalized bbox must land on
            # the same pixel box (no cumulative rescaling).
            assert second["region_pixels"] == first["region_pixels"], (width, height)

    def test_clip_flag_when_out_of_bounds(self):
        result = CT.transform([0, 0, 1200, 1200], 500, 500)
        assert result["clipped"] is True
        assert result["region_pixels"][2] <= 500
        assert result["region_pixels"][3] <= 500

    def test_normalized_request_preserved_when_clipped(self):
        """The model's requested bbox stays recorded even if pixels are clipped."""
        result = CT.transform([0, 0, 1500, 1500], 400, 400)
        assert result["region_normalized_1000"] == [0, 0, 1500, 1500]
        assert result["region_pixels"] == [0, 0, 400, 400]

    def test_full_image_request(self):
        result = CT.transform([0, 0, 1000, 1000], 512, 256)
        assert result["region_pixels"] == [0, 0, 256, 512]
        assert result["clipped"] is False
