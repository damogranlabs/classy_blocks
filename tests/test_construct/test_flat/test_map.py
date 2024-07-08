import unittest

import numpy as np

from classy_blocks.construct.flat.map import QuadMap
from classy_blocks.construct.flat.quad import Quad
from classy_blocks.util import functions as f


class QuadTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    @property
    def quad(self) -> Quad:
        return Quad(self.positions, [0, 1, 2, 3])

    def test_quad_update(self):
        quad = self.quad
        new_position = [-1, -1, 0]

        self.positions[0] = new_position
        quad.update(self.positions)

        np.testing.assert_equal(quad.points, self.positions)

    def test_contains(self):
        self.assertTrue(self.quad.contains(f.vector(0, 0, 0)))

    def test_not_contains(self):
        self.assertFalse(self.quad.contains(f.vector(0, 0, 1)))

    def test_perimeter_square(self):
        self.assertEqual(self.quad.perimeter, 4)

    def test_perimeter_rectangle(self):
        delta = f.vector(1, 0, 0)
        self.positions[1] += delta
        self.positions[2] += delta

        self.quad.update(self.positions)

        self.assertEqual(self.quad.perimeter, 6)

    def test_center(self):
        np.testing.assert_equal(self.quad.center, [0.5, 0.5, 0])

    def test_e1(self):
        np.testing.assert_almost_equal(self.quad.e1, [1, 0, 0])

    def test_e2(self):
        np.testing.assert_almost_equal(self.quad.e2, [0, 1, 0])

    def test_normal(self):
        np.testing.assert_almost_equal(self.quad.normal, [0, 0, 1])


class QuadMapTests(unittest.TestCase):
    @property
    def positions(self):
        return np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1.5, 1.5, 0],  # a moved vertex
                [2, 1, 0],
                [0, 2, 0],
                [1, 2, 0],
                [2, 2, 0],
            ]
        )

    @property
    def quads(self):
        return [
            (0, 1, 4, 3),
            (1, 2, 5, 4),
            (3, 4, 7, 6),
            (4, 5, 8, 7),
        ]

    @property
    def quad_map(self):
        return QuadMap(self.positions, self.quads)

    def test_positions(self):
        np.testing.assert_equal(self.quad_map.positions, self.positions)

    def test_find_neighbours(self):
        # A random blocking (quadding)
        positions = np.zeros((9, 3))
        indexes = [(1, 2, 7, 6), (2, 3, 4, 7), (7, 4, 5, 6), (0, 1, 6, 8)]

        quad_map = QuadMap(positions, indexes)

        self.assertDictEqual(
            quad_map.neighbours,
            {
                0: {8, 1},
                1: {0, 2, 6},
                2: {1, 3, 7},
                3: {2, 4},
                4: {3, 5, 7},
                5: {4, 6},
                6: {8, 1, 5, 7},
                7: {2, 4, 6},
                8: {0, 6},
            },
        )

    def test_fixed_points(self):
        # Monocylinder, core is quads[0]
        positions = np.zeros((8, 3))
        indexes = [
            (0, 1, 2, 3),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ]

        quad_map = QuadMap(positions, indexes)

        fixed_points = quad_map.boundary_points

        self.assertSetEqual(
            fixed_points,
            {4, 5, 6, 7},
        )

    def test_update(self):
        positions = self.positions
        positions[0] = [0.5, 0.5, 0]

        quad_map = self.quad_map
        quad_map.update(positions)

        np.testing.assert_equal(quad_map.quads[0].points[0], [0.5, 0.5, 0])

    def test_smooth(self):
        # a grid of vertices 3x3
        quad_map = QuadMap(self.positions, self.quads)
        for _ in range(10):
            quad_map.smooth_laplacian()

        np.testing.assert_almost_equal(quad_map.positions[4], [1, 1, 0], decimal=5)
