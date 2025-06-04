import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.base.exceptions import PointCreationError
from classy_blocks.construct.point import Point
from classy_blocks.util.constants import TOL


class PointTests(unittest.TestCase):
    """Border cases for Point object"""

    @property
    def point(self) -> Point:
        """Test subject"""
        return Point([1, 1, 1])

    @parameterized.expand(
        [
            ((1, 1),),  # To few arguments!
            ((1, 1, 1, 1),),  # To much arguments!"
        ]
    )
    def test_invalid_creation_parameters(self, position):
        with self.assertRaises(PointCreationError):
            Point(list(position))

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

    def test_project_double(self):
        point = self.point

        point.project("terrain")
        point.project("terrain")

        self.assertEqual(len(point.projected_to), 1)

    def test_project_twice(self):
        """Multiple calls to project() must add geometry to the projections list"""
        point = self.point

        point.project("terrain")
        point.project("also_terrain")

        self.assertListEqual(point.projected_to, ["terrain", "also_terrain"])

    def test_project_twice_mixed(self):
        point = self.point

        point.project("terrain")
        point.project(["also_terrain", "also_also_terrain"])

        self.assertListEqual(point.projected_to, ["terrain", "also_terrain", "also_also_terrain"])

    def test_mirror_default_origin(self):
        point = self.point
        point.mirror([1, 1, 1])

        np.testing.assert_almost_equal(point.position, [-1, -1, -1])
