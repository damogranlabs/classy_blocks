import unittest
from typing import get_args

from classy_blocks.construct.operations.box import Box
from classy_blocks.mesh import Mesh
from classy_blocks.modify.find.geometric import GeometricFinder
from classy_blocks.optimize.grid import Grid
from classy_blocks.types import AxisType


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

    def get_grid(self, mesh: Mesh) -> Grid:
        return Grid.from_mesh(mesh)
