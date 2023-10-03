import unittest
from math import cos, sin

import numpy as np
from parameterized import parameterized

from classy_blocks.base.curve import Curve
from classy_blocks.types import NPPointType


class CircleCurveTests(unittest.TestCase):
    def setUp(self):
        def circle(t: float) -> NPPointType:
            # Test curve: a circle with radius 1
            return np.array([cos(t), sin(t), 0])

        self.circle = circle
        self.curve = Curve(circle, [0, np.pi])
        self.places = int(np.log10(self.curve.eps))

    def test_arc_length_halfcircle(self):
        self.assertAlmostEqual(self.curve.get_length(0, np.pi), np.pi, places=self.places)

    def test_arc_length_quartercircle(self):
        self.assertAlmostEqual(self.curve.get_length(0, np.pi / 2), np.pi / 2, places=self.places)

    def test_length_property(self):
        self.assertAlmostEqual(self.curve.length, np.pi, places=self.places)

    @parameterized.expand(
        (
            ([1, 0, 0], 0),
            ([0, 1, 0], np.pi / 2),
            ([-1, 0, 0], np.pi),
        )
    )
    def test_closest_param(self, position, param):
        self.assertAlmostEqual(self.curve.get_closest_param(position), param)

    def test_from_points(self):
        points = [self.circle(t) for t in np.linspace(0, np.pi)]
        curve = Curve.from_points(points, 0, np.pi)

        self.assertAlmostEqual(curve.length, np.pi, places=self.places)

    def test_discretize(self):
        count = 20

        discretized_points = self.curve.discretize(count=count)
        analytic_points = [self.circle(t) for t in np.linspace(0, np.pi, count)]

        np.testing.assert_almost_equal(discretized_points, analytic_points)
