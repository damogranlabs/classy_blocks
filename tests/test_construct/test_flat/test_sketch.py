import unittest

import numpy as np

from classy_blocks.construct.flat.quad import find_neighbours, get_fixed_points
from classy_blocks.construct.flat.sketch import MappedSketch
from classy_blocks.construct.flat.sketches.grid import Grid


class MappedSketchTests(unittest.TestCase):
    @property
    def positions(self):
        return [
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

    @property
    def quads(self):
        return [
            (0, 1, 4, 3),
            (1, 2, 5, 4),
            (3, 4, 7, 6),
            (4, 5, 8, 7),
        ]

    @property
    def sketch(self):
        return MappedSketch(self.positions, self.quads)

    def test_find_neighbours(self):
        # A random blocking (quadding)
        quads = [(1, 2, 7, 6), (2, 3, 4, 7), (7, 4, 5, 6), (0, 1, 6, 8)]

        neighbours = find_neighbours(quads)

        self.assertDictEqual(
            neighbours,
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
        quads = [
            (0, 1, 2, 3),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ]

        fixed_points = get_fixed_points(quads)

        self.assertSetEqual(
            fixed_points,
            {4, 5, 6, 7},
        )

    def test_smooth(self):
        # a grid of vertices 3x3

        sketch = MappedSketch(self.positions, self.quads, smooth_iter=10)

        np.testing.assert_almost_equal(sketch.faces[0].point_array[2], [1, 1, 0])

    def test_faces(self):
        self.assertEqual(len(self.sketch.faces), 4)

    def test_grid(self):
        self.assertEqual(len(self.sketch.grid), 1)

    def test_center(self):
        sketch = MappedSketch(self.positions, self.quads, smooth_iter=10)

        np.testing.assert_almost_equal(sketch.center, [1, 1, 0])


class GridSketchTests(unittest.TestCase):
    def test_construct(self):
        grid = Grid([0, 0, 0], [1, 1, 0], 3, 3)

        self.assertEqual(len(grid.faces), 9)

    def test_center(self):
        grid = Grid([0, 0, 0], [1, 1, 0], 3, 3)

        np.testing.assert_almost_equal(grid.center, [0.5, 0.5, 0])
