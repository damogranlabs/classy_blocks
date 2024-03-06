import unittest
from math import cos, sin

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.curves.analytic import AnalyticCurve, CircleCurve, LineCurve
from classy_blocks.types import NPPointType
from classy_blocks.util import functions as f
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
        # use less strict checking because of discretization error
        self.assertAlmostEqual(self.curve.get_length(0, np.pi), self.radius * np.pi, places=2)

    def test_arc_length_quartercircle(self):
        self.assertAlmostEqual(self.curve.get_length(0, np.pi / 2), self.radius * np.pi / 2, places=2)

    def test_length_property(self):
        self.assertAlmostEqual(self.curve.length, self.radius * np.pi, places=2)

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


class LineCurveTests(unittest.TestCase):
    def setUp(self):
        self.point_1 = [0, 0, 0]
        self.point_2 = [1, 1, 0]

        self.bounds = (0, 1)

    @property
    def curve(self) -> LineCurve:
        return LineCurve(self.point_1, self.point_2, self.bounds)

    def test_init(self):
        _ = self.curve

    @parameterized.expand(
        (
            (0, [0, 0, 0]),
            (0.5, [0.5, 0.5, 0]),
            (1, [1, 1, 0]),
        )
    )
    def test_values_within_bounds(self, param, value):
        np.testing.assert_almost_equal(self.curve.get_point(param), value)

    @parameterized.expand(
        (
            (-1,),
            (2,),
        )
    )
    def test_values_outside_bounds(self, param):
        with self.assertRaises(ValueError):
            _ = self.curve.get_point(param)

    def test_translate(self):
        curve = self.curve
        curve.translate([1, 0, 0])

        np.testing.assert_almost_equal(curve.get_point(0), [1, 0, 0])

    def test_rotate(self):
        curve = self.curve
        curve.rotate(np.pi / 4, [0, 0, 1], [0, 0, 0])

        np.testing.assert_almost_equal(curve.get_point(1), [0, 2**0.5, 0])

    def test_scale(self):
        curve = self.curve
        curve.scale(2, [0, 0, 0])

        np.testing.assert_almost_equal(curve.get_point(1), [2, 2, 0])

    def test_scale_nocenter(self):
        curve = self.curve
        curve.scale(2)

        np.testing.assert_almost_equal(curve.get_point(0), [-0.5, -0.5, 0])

    def test_center(self):
        np.testing.assert_almost_equal(self.curve.center, [0.5, 0.5, 0])


class CircleCurveTests(unittest.TestCase):
    def setUp(self):
        self.origin = [1, 1, 0]
        self.rim = [2, 1, 0]
        self.normal = [0, 0, 1]

        self.bounds = (0, 2 * np.pi)

    @property
    def curve(self) -> CircleCurve:
        return CircleCurve(self.origin, self.rim, self.normal, self.bounds)

    @parameterized.expand(((0, [2, 1, 0]), (np.pi / 2, [1, 2, 0]), (np.pi, [0, 1, 0]), (3 * np.pi / 2, [1, 0, 0])))
    def test_circle_points(self, param, point):
        np.testing.assert_almost_equal(self.curve.get_point(param), point)

    def test_translate(self):
        curve = self.curve
        curve.translate([-1, 0, 0])

        np.testing.assert_almost_equal(curve.get_point(0), [1, 1, 0])

    def test_rotate(self):
        curve = self.curve
        curve.rotate(np.pi / 2, [0, 0, 1])

        np.testing.assert_almost_equal(curve.get_point(0), [1, 2, 0])

    def test_scale(self):
        curve = self.curve
        curve.scale(2)

        np.testing.assert_almost_equal(f.norm(curve.get_point(0) - curve.center), 2)
