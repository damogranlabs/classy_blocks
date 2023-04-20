import unittest

import numpy as np

from classy_blocks.construct.point import Point
from classy_blocks.util.constants import TOL


class PointTests(unittest.TestCase):
    """Border cases for Point object"""

    @property
    def point(self) -> Point:
        """Test subject"""
        return Point([1, 1, 1])

    def test_default_rotate_origin(self):
        """Rotation without an origin"""
        np.testing.assert_array_almost_equal(self.point.rotate(np.pi, [0, 0, 1]).position, [-1, -1, 1])

    def test_default_scale_origin(self):
        """Rotation without an origin"""
        np.testing.assert_array_almost_equal(self.point.scale(2).position, [2, 2, 2])

    def test_center(self):
        """Center property"""
        np.testing.assert_array_equal(self.point.center, self.point.position)

    def test_points_equal(self):
        """Points are equal when they are close enough"""
        delta = TOL / 10
        other = Point([1 + delta, 1 + delta, 1 + delta])

        self.assertTrue(self.point == other)

    def test_points_not_equal(self):
        """Points are not equal when they are more than TOL apart"""
        delta = 2 * TOL
        other = Point([1 + delta, 1 + delta, 1 + delta])

        self.assertFalse(self.point == other)
