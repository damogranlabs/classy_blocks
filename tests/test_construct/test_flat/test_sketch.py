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

    def test_update(self):
        sketch = MappedSketch(self.positions, self.quads)

        updated_positions = self.positions
        updated_positions[0] = [0.1, 0.1, 0.1]
        sketch.update(updated_positions)

        np.testing.assert_equal(sketch.faces[0].point_array[0], [0.1, 0.1, 0.1])

    def test_merge(self):
        sketch_1_pos = self.positions[:6]
        sketch_2_pos = self.positions[3:]
        sketch_2_quad = (np.asarray(self.quads[2:])-3).tolist()

        sketch_1 = MappedSketch(sketch_1_pos, self.quads[:2])
        sketch_2 = MappedSketch(sketch_2_pos, sketch_2_quad)
        sketch_1.merge(sketch_2)

        np.testing.assert_equal(np.asarray([face.point_array for face in sketch_1.faces]),
                                np.asarray([face.point_array for face in self.sketch.faces]))
        self.assertEqual(sketch_1.indexes, self.sketch.indexes)


class GridSketchTests(unittest.TestCase):
    def test_construct(self):
        grid = Grid([0, 0, 0], [1, 1, 0], 3, 3)

        self.assertEqual(len(grid.faces), 9)

    def test_center(self):
        grid = Grid([0, 0, 0], [1, 1, 0], 3, 3)

        np.testing.assert_almost_equal(grid.center, [0.5, 0.5, 0])
