import unittest

import numpy as np

from classy_blocks.construct.curves.analytic import AnalyticCurve, CircleCurve, LineCurve
from classy_blocks.construct.curves.discrete import DiscreteCurve
from classy_blocks.construct.curves.interpolated import LinearInterpolatedCurve, SplineInterpolatedCurve


class VertexCuspTests(unittest.TestCase):
    def setUp(self):
        # right-angle corner at the middle point
        self.corner = [[0, 0, 0], [1, 0, 0], [1, 1, 0]]

    def test_discrete_single_cusp(self):
        cusps = DiscreteCurve(self.corner).find_cusps(0.1)
        self.assertEqual(len(cusps), 1)

    def test_discrete_cusp_param_is_index(self):
        cusp = DiscreteCurve(self.corner).find_cusps(0.1)[0]
        self.assertEqual(cusp.param, 1)

    def test_discrete_cusp_point_is_the_corner(self):
        cusp = DiscreteCurve(self.corner).find_cusps(0.1)[0]
        np.testing.assert_array_equal(cusp.point, [1, 0, 0])

    def test_discrete_cusp_angle_is_right_angle(self):
        cusp = DiscreteCurve(self.corner).find_cusps(0.1)[0]
        self.assertAlmostEqual(cusp.angle, np.pi / 2)

    def test_linear_cusp_param_is_normalized(self):
        # LinearInterpolatedCurve bounds are (0, 1): middle of 3 points -> 0.5
        cusp = LinearInterpolatedCurve(self.corner).find_cusps(0.1)[0]
        self.assertAlmostEqual(cusp.param, 0.5)

    def test_linear_cusp_point_is_the_corner(self):
        cusp = LinearInterpolatedCurve(self.corner).find_cusps(0.1)[0]
        np.testing.assert_array_almost_equal(cusp.point, [1, 0, 0])

    def test_linear_cusp_param_matches_point_on_unequal_segments(self):
        # equalize=True (default) parameterizes by chord length, not i/segments;
        # the reported param must actually evaluate to the reported point
        curve = LinearInterpolatedCurve([[0, 0, 0], [1, 0, 0], [1, 3, 0]])
        cusp = curve.find_cusps(0.1)[0]
        np.testing.assert_array_almost_equal(curve.get_point(cusp.param), cusp.point)

    def test_straight_polyline_has_no_cusps(self):
        straight = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
        self.assertEqual(DiscreteCurve(straight).find_cusps(0.1), [])

    def test_two_point_curve_has_no_cusps(self):
        self.assertEqual(DiscreteCurve([[0, 0, 0], [1, 0, 0]]).find_cusps(0.1), [])

    def test_multiple_cusps_in_curve_order(self):
        zigzag = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [2, 1, 0]]
        cusps = DiscreteCurve(zigzag).find_cusps(0.1)
        self.assertEqual([c.param for c in cusps], [1, 2])

    def test_threshold_filters_shallow_bends(self):
        gentle = [[0, 0, 0], [1, 0, 0], [2, 0.01, 0]]
        self.assertEqual(DiscreteCurve(gentle).find_cusps(0.1), [])


class SmoothCurveCuspTests(unittest.TestCase):
    def test_analytic_curve_raises(self):
        curve = AnalyticCurve(lambda t: np.array([t, t**2, 0]), (0, 1))
        with self.assertRaises(NotImplementedError):
            curve.find_cusps(0.1)

    def test_circle_curve_raises(self):
        curve = CircleCurve([0, 0, 0], [1, 0, 0], [0, 0, 1])
        with self.assertRaises(NotImplementedError):
            curve.find_cusps(0.1)

    def test_line_curve_raises(self):
        curve = LineCurve([0, 0, 0], [1, 1, 0])
        with self.assertRaises(NotImplementedError):
            curve.find_cusps(0.1)

    def test_spline_curve_warns(self):
        curve = SplineInterpolatedCurve([[0, 0, 0], [1, 1, 0], [2, 0, 0], [3, 1, 0]])
        with self.assertWarns(Warning):
            curve.find_cusps(0.1)
