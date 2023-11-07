import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.curves.interpolators import LinearInterpolator
from classy_blocks.construct.point import Point


class LinearInterpolatorTests(unittest.TestCase):
    def setUp(self):
        # a simple square wave
        self.points = [
            Point([0, 0, 0]),
            Point([0, 1, 0]),
            Point([1, 1, 0]),
            Point([1, 0, 0]),
            Point([2, 0, 0]),
        ]

    @parameterized.expand(
        (
            (0, [0, 0, 0]),
            (1 / 4, [0, 1, 0]),
            (2 / 4, [1, 1, 0]),
            (0.5 / 4, [0, 0.5, 0]),
        )
    )
    def test_points(self, param, result):
        intfun = LinearInterpolator(self.points, True)

        np.testing.assert_almost_equal(intfun(param), result)

    def test_extrapolate_exception(self):
        intfun = LinearInterpolator(self.points, False)

        with self.assertRaises(ValueError):
            _ = intfun(-1)

    def test_cache(self):
        intfun = LinearInterpolator(self.points, True)

        for point in self.points:
            point.rotate(np.pi / 2, [0, 0, 1])

        # the interpolation function must not change unless invalidated
        np.testing.assert_equal(intfun(0.25), [0, 1, 0])

    def test_invalidate(self):
        intfun = LinearInterpolator(self.points, True)

        for point in self.points:
            point.translate([0, 0, 1])

        intfun.invalidate()

        np.testing.assert_equal(intfun(0.25), [0, 1, 1])
