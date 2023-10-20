import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.curves.interpolated import LinearInterpolatedCurve, SplineInterpolatedCurve


class LinearInterpolatedCurveTests(unittest.TestCase):
    def setUp(self):
        # a simple square wave
        self.points = [
            [0, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
            [2, 0, 0],
        ]

    @property
    def curve(self) -> LinearInterpolatedCurve:
        return LinearInterpolatedCurve(self.points, True)

    def test_discretize(self):
        np.testing.assert_array_equal(self.curve.discretize(count=len(self.points)), self.points)

    @parameterized.expand(
        (
            (0, 1),
            (0, 2),
            (0, 3),
            (2, 4),
            (3, 4),
            (0, 4),
        )
    )
    def test_get_length(self, param_from, param_to):
        length = param_to - param_from
        self.assertAlmostEqual(self.curve.get_length(param_from, param_to), length)

    def test_length(self):
        self.assertEqual(self.curve.length, 4)

    def test_get_point(self):
        np.testing.assert_array_equal(self.curve.get_point(0.5), [0, 0.5, 0])

    @parameterized.expand(
        (
            # without starting param
            ([-1, 3, 0], None, 1),
            ([0.2, 2, 0], None, 1.2),
            ([0.8, 2, 0], None, 1.8),
            ([1.1, -0.5, 0], None, 3.1),
            # with starting param: same point will yield different params
            ([0.5, 0.5, 0], 0.3, 0.5),
            ([0.5, 0.5, 0], 1.3, 1.5),
            ([0.5, 0.5, 0], 2.3, 2.5),
        )
    )
    def test_get_closest_param(self, point, start, param):
        self.assertAlmostEqual(self.curve.get_closest_param(point, start), param, places=3)

    def test_transform(self):
        curve = self.curve

        curve.translate([1, 1, 1])

        np.testing.assert_array_equal(curve.get_point(0), [1, 1, 1])


class SplineInterpolatedCurveTests(unittest.TestCase):
    def setUp(self):
        # a simple square wave
        self.points = np.array(
            [
                [0, 0, 0],
                [1, 1, 0],
                [2, 0, 0],
                [3, -1, 0],
                [4, 0, 0],
            ]
        )

    @property
    def curve(self) -> SplineInterpolatedCurve:
        return SplineInterpolatedCurve(self.points, True)

    def test_length(self):
        # spline must be longer than linear segments combined
        self.assertGreater(self.curve.length, 4 * 2**0.5)
        self.assertLess(self.curve.length, 8)

    @parameterized.expand(
        (
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
        )
    )
    def test_through_points(self, param):
        """Make sure the curve goes through original points"""
        # param equals to index
        np.testing.assert_almost_equal(self.curve.get_point(param), self.points[param])

    def test_transform(self):
        curve = self.curve

        curve.translate([0, 0, 1])

        np.testing.assert_equal(curve.get_point(1), [1, 1, 1])
