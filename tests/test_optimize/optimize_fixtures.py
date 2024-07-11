import unittest
from typing import get_args

import numpy as np

from classy_blocks.construct.operations.box import Box
from classy_blocks.mesh import Mesh
from classy_blocks.modify.find.geometric import GeometricFinder
from classy_blocks.optimize.grid import HexGrid, QuadGrid
from classy_blocks.types import AxisType


class SketchTestsBase(unittest.TestCase):
    @property
    def positions(self):
        return np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [0, 1, 0],
                [1.5, 1.5, 0],  # a moved vertex, should be [1, 1, 0]
                [2, 1, 0],
                [0, 2, 0],
                [1, 2, 0],
                [2, 2, 0],
            ]
        )

    @property
    def quads(self):
        return [
            [0, 1, 4, 3],
            [1, 2, 5, 4],
            [3, 4, 7, 6],
            [4, 5, 8, 7],
        ]

    @property
    def grid(self):
        return QuadGrid(self.positions, self.quads)


class BoxTestsBase(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh()

        # generate a cube, consisting of 2x2x2 smaller cubes
        for x in (-1, 0):
            for y in (-1, 0):
                for z in (-1, 0):
                    box = Box([x, y, z], [x + 1, y + 1, z + 1])

                    for axis in get_args(AxisType):
                        box.chop(axis, count=10)

                    self.mesh.add(box)

        self.mesh.assemble()

        self.finder = GeometricFinder(self.mesh)

    def get_vertex(self, position):
        return next(iter(self.finder.find_in_sphere(position)))

    def get_grid(self, mesh: Mesh) -> HexGrid:
        return HexGrid.from_mesh(mesh)
