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
