import unittest
from math import cos, sin

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.curves.analytic import AnalyticCurve
from classy_blocks.types import NPPointType
from classy_blocks.util.constants import TOL


class AnalyticCurveTests(unittest.TestCase):
    def setUp(self):
        self.radius = 3

        def circle(t: float) -> NPPointType:
            # Test curve: a circle with radius 1
            return np.array([self.radius * cos(t), self.radius * sin(t), 0])

        self.circle = circle
        self.curve = AnalyticCurve(circle, (0, np.pi))
        self.places = int(np.log10(1 / TOL)) - 1

    def test_arc_length_halfcircle(self):
        self.assertAlmostEqual(self.curve.get_length(0, np.pi), self.radius * np.pi, places=self.places)

    def test_arc_length_quartercircle(self):
        self.assertAlmostEqual(self.curve.get_length(0, np.pi / 2), self.radius * np.pi / 2, places=self.places)

    def test_length_property(self):
        self.assertAlmostEqual(self.curve.length, self.radius * np.pi, places=self.places)

    @parameterized.expand(
        (
            ([3, 0, 0], 0),
            ([0, 3, 0], np.pi / 2),
            ([-3, 0, 0], np.pi),
        )
    )
    def test_closest_param(self, position, param):
        self.assertAlmostEqual(self.curve.get_closest_param(position), param, places=5)

    def test_discretize(self):
        count = 20

        discretized_points = self.curve.discretize(count=count)
        analytic_points = [self.circle(t) for t in np.linspace(0, np.pi, count)]

        np.testing.assert_almost_equal(discretized_points, analytic_points)

    def test_transform(self):
        """Raise an error when trying to transform an analytic curve"""
        with self.assertRaises(NotImplementedError):
            self.curve.translate([1, 1, 1])
