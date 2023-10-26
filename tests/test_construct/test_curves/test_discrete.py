import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.curves.discrete import DiscreteCurve


class DiscreteCurveTests(unittest.TestCase):
    def setUp(self):
        self.points = [
            [0, 0, 0],
            [1, 1, 0],
            [2, 4, 0],
            [3, 9, 0],
        ]
        self.segment_lengths = np.array([2**0.5, 10**0.5, 26**0.5])

    @property
    def curve(self) -> DiscreteCurve:
        return DiscreteCurve(self.points)

    def test_single_point(self):
        """Only one point was provided"""
        with self.assertRaises(ValueError):
            _ = DiscreteCurve([[0, 0, 0]])

    def test_wrong_shape(self):
        """Points are not in 3-dimensions"""
        with self.assertRaises(ValueError):
            _ = DiscreteCurve([[0, 0], [1, 0]])

    @parameterized.expand(((-1, 1), (0, 5), (0, 0)))
    def test_discretize_wrong_params(self, param_from, param_to):
        """Invalid params passed to discretize() method"""
        with self.assertRaises(ValueError):
            self.curve.discretize(param_from, param_to)

    def test_discretize(self):
        """Call discretize() without params"""
        np.testing.assert_equal(self.curve.discretize(), self.points)

    def test_discretize_partial(self):
        """Discretize with given params"""
        np.testing.assert_equal(self.curve.discretize(1, 3), self.points[1:4])

    def test_discretize_inverted(self):
        """Discretize with param_from bigger than param_to"""
        discretized = self.curve.discretize(3, 1)
        expected = np.flip(self.points[1:4], axis=0)

        np.testing.assert_equal(discretized, expected)

    def test_length(self):
        self.assertEqual(self.curve.length, sum(self.segment_lengths))

    @parameterized.expand(
        (
            (0, 1),
            (0, 2),
            (0, 3),
        )
    )
    def test_get_length(self, param_from, param_to):
        """Length of single segment"""
        length = sum(self.segment_lengths[param_from:param_to])

        self.assertEqual(self.curve.get_length(param_from, param_to), length)

    @parameterized.expand(
        (
            ([0, -1, 0], 0),
            ([0.8, 0.8, 0], 1),
            ([1.6, 3.5, 0], 2),
            ([10, 10, 0], 3),
        )
    )
    def test_get_closest_param(self, point, index):
        self.assertEqual(self.curve.get_closest_param(point), index)

    def test_transform(self):
        """A simple translation to test .parts property"""
        curve = self.curve
        curve.translate([0, 0, 1])

        np.testing.assert_equal(curve.get_point(0), [0, 0, 1])

    def test_center(self):
        np.testing.assert_almost_equal(self.curve.center, [1.5, 3.5, 0])

    def test_scale(self):
        scaled = self.curve.scale(2, [0, 0, 0])

        np.testing.assert_array_almost_equal(
            scaled.discretize(),
            [
                [0, 0, 0],
                [2, 2, 0],
                [4, 8, 0],
                [6, 18, 0],
            ],
        )
