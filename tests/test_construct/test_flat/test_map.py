import unittest

import numpy as np

from classy_blocks.construct.flat.map import QuadMap


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

        fixed_points = quad_map.fixed_points

        self.assertSetEqual(
            fixed_points,
            {4, 5, 6, 7},
        )

    def test_smooth(self):
        # a grid of vertices 3x3
        quad_map = QuadMap(self.positions, self.quads)
        quad_map.smooth_laplacian(10)

        np.testing.assert_almost_equal(quad_map.positions[4], [1, 1, 0])
