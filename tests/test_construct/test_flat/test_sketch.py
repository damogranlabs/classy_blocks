import unittest

import numpy as np

from classy_blocks.construct.flat.sketches.grid import Grid
from classy_blocks.construct.flat.sketches.mapped import MappedSketch


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
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ]

    @property
    def sketch(self):
        return MappedSketch(self.positions, self.quads)

    def test_faces(self):
        self.assertEqual(len(self.sketch.faces), 4)

    def test_grid(self):
        self.assertEqual(len(self.sketch.grid), 1)


class GridSketchTests(unittest.TestCase):
    def test_construct(self):
        grid = Grid([0, 0, 0], [1, 1, 0], 3, 3)

        self.assertEqual(len(grid.faces), 9)

    def test_center(self):
        grid = Grid([0, 0, 0], [1, 1, 0], 3, 3)

        np.testing.assert_almost_equal(grid.center, [0.5, 0.5, 0])
