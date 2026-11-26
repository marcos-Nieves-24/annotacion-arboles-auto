"""Deterministic tests for centroid target generation.

Run: .venv-verify/bin/python -m pytest scripts/test_build_centroid_targets.py -q
"""
import math

import numpy as np
import pytest

from build_centroid_targets import polygon_centroid, equivalent_radius, build_targets


def test_polygon_centroid_square():
    cx, cy = polygon_centroid([[100, 100, 200, 100, 200, 200, 100, 200]])
    assert (cx, cy) == pytest.approx((150.0, 150.0), abs=1e-6)


def test_polygon_centroid_triangle():
    cx, cy = polygon_centroid([[0, 0, 30, 0, 0, 30]])
    assert (cx, cy) == pytest.approx((10.0, 10.0), abs=1e-6)


def test_equivalent_radius_matches_plan_formula():
    assert equivalent_radius(math.pi * 51.4 ** 2) == pytest.approx(51.4, abs=1e-6)


def test_build_targets_peak_and_radius():
    # One crown at (257, 257) with radius ~50 px -> heatmap peak near (64, 64).
    anns = [{
        "segmentation": [[207, 257, 257, 207, 307, 257, 257, 307]],
        "area": math.pi * 50 ** 2,
        "iscrowd": 0,
    }]
    heat, radius, weight, centers = build_targets(anns, size=512, stride=4)
    assert heat.shape == (128, 128)
    assert radius.shape == (128, 128)
    assert weight.shape == (128, 128)
    assert len(centers) == 1
    assert (centers[0][0], centers[0][1]) == pytest.approx((257.0, 257.0), abs=1e-6)
    ys, xs = np.nonzero(heat)
    assert (xs.mean(), ys.mean()) == pytest.approx((64.25, 64.25), abs=1)
    iy, ix = int(round(ys.mean())), int(round(xs.mean()))
    assert heat[iy, ix] == pytest.approx(1.0, abs=0.02)
    assert radius[iy, ix] == pytest.approx(50.0, abs=1e-3)
    assert weight[iy, ix] == 1.0


def test_build_targets_two_crowns_use_max_not_sum():
    anns = [
        {"segmentation": [[100, 100, 140, 100, 140, 140, 100, 140]],
         "area": math.pi * 20 ** 2, "iscrowd": 0},
        {"segmentation": [[120, 120, 160, 120, 160, 160, 120, 160]],
         "area": math.pi * 20 ** 2, "iscrowd": 0},
    ]
    heat, radius, weight, centers = build_targets(anns, size=512, stride=4)
    assert len(centers) == 2
    assert heat.max() <= 1.0 + 1e-6
    assert heat.sum() < 2.0 * heat.max() * 50


def test_build_targets_border_crown_downweighted():
    anns = [{
        "segmentation": [[0, 100, 40, 100, 40, 140, 0, 140]],
        "area": math.pi * 20 ** 2, "iscrowd": 0,
    }]
    heat, radius, weight, centers = build_targets(anns, size=512, stride=4)
    iy, ix = heat.argmax() // 128, heat.argmax() % 128
    assert weight[iy, ix] == 0.5


def test_build_targets_empty_image():
    heat, radius, weight, centers = build_targets([], size=512, stride=4)
    assert heat.max() == 0.0
    assert centers == []
