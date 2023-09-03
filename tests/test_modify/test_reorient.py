import unittest

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.operations.loft import Loft
from classy_blocks.modify.reorient.viewpoint import Quadrangle, Triangle, ViewpointReorienter
from classy_blocks.util import functions as f
from tests.fixtures.block import BlockTestCase


class TriangleTests(unittest.TestCase):
    def setUp(self):
        self.triangle = Triangle([f.vector(0, 0, 0), f.vector(1, 0, 0), f.vector(0, 1, 0)])

    def test_normal(self):
        np.testing.assert_array_equal(self.triangle.normal, [0, 0, 1])

    def test_center(self):
        np.testing.assert_array_almost_equal(self.triangle.center, [1 / 3, 1 / 3, 0])

    def test_flip(self):
        self.triangle.flip()

        np.testing.assert_array_equal(self.triangle.normal, [0, 0, -1])

    def test_orient_no_change(self):
        self.triangle.orient(f.vector(-0.5, -0.5, -0.5))

        np.testing.assert_array_almost_equal(self.triangle.normal, [0, 0, 1])

    def test_orient_change(self):
        self.triangle.orient(f.vector(0.5, 0.5, 0.5))

        np.testing.assert_array_almost_equal(self.triangle.normal, [0, 0, -1])


class QuadrangleTests(unittest.TestCase):
    def setUp(self):
        super().setUp()

        # lower-left part of quad
        self.tri_1 = Triangle([f.vector(0, 0, 0), f.vector(1, 0, 0), f.vector(1, 1, 0)])

        # upper-right part
        self.tri_2 = Triangle([f.vector(0, 0, 0), f.vector(1, 1, 0), f.vector(0, 1, 0)])

    @property
    def quad(self) -> Quadrangle:
        return Quadrangle([self.tri_1, self.tri_2])

    def test_get_common_points(self):
        list_1 = self.tri_1.points
        list_2 = self.tri_2.points

        common_points = Quadrangle.get_common_points(list_1, list_2)

        np.testing.assert_array_equal(common_points, [[0, 0, 0], [1, 1, 0]])

    def test_get_unique_points(self):
        list_1 = self.tri_1.points
        list_2 = self.tri_2.points

        unique_points = Quadrangle.get_unique_points(list_1, list_2)

        np.testing.assert_array_almost_equal(unique_points, [[1, 0, 0], [0, 1, 0]])

    def test_point_count(self):
        quad = Quadrangle([self.tri_1, self.tri_2])

        self.assertEqual(len(quad.points), 4)


class ViewpointReorienterTests(BlockTestCase):
    def setUp(self):
        self.reorienter = ViewpointReorienter([0.5, -10, 0.5], [0.5, 0.5, 10])

    @property
    def loft(self) -> Loft:
        return self.make_loft(0)

    @parameterized.expand(
        (
            (
                "x",
                np.pi / 2,
            ),
            (
                "x",
                np.pi,
            ),
            (
                "x",
                3 * np.pi / 2,
            ),
            (
                "y",
                np.pi / 2,
            ),
            (
                "y",
                np.pi,
            ),
            (
                "y",
                3 * np.pi / 2,
            ),
            (
                "z",
                np.pi / 2,
            ),
            (
                "z",
                np.pi,
            ),
            (
                "z",
                3 * np.pi / 2,
            ),
        )
    )
    def test_sort_regular(self, axis, angle):
        axis = {
            "x": [1, 0, 0],
            "y": [0, 1, 0],
            "z": [0, 0, 1],
        }[axis]

        loft = self.loft

        loft.rotate(angle, axis, self.loft.center)
        self.reorienter.reorient(loft)

        np.testing.assert_array_almost_equal(
            loft.point_array,
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        )

    def test_translated_face(self):
        loft = self.loft
        original_face = loft.top_face.copy()
        loft.top_face.translate([1, 1, 0])

        # make sure top face is still on top
        self.reorienter.reorient(loft)

        np.testing.assert_array_equal(loft.top_face.point_array, original_face.translate([1, 1, 0]).point_array)

    def test_rotated_face(self):
        loft = self.loft
        center = loft.center
        original_face = loft.top_face.copy()
        loft.top_face.rotate(0.9 * np.pi / 4, [0, 0, 1], center)

        # make sure top face is still on top
        self.reorienter.reorient(loft)

        np.testing.assert_array_equal(
            loft.top_face.point_array, original_face.rotate(0.9 * np.pi / 4, [0, 0, 1], center).point_array
        )

    def test_moved_point(self):
        loft = self.loft
        loft.top_face.points[2].translate([2, 2, 2])

        self.reorienter.reorient(loft)

        np.testing.assert_array_almost_equal(loft.top_face.point_array, [[0, 0, 1], [1, 0, 1], [3, 3, 3], [0, 1, 1]])
