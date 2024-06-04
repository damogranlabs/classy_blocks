import unittest

from classy_blocks.construct.flat.sketches.grid import Grid
from classy_blocks.construct.stack import ExtrudedStack
from classy_blocks.mesh import Mesh
from classy_blocks.modify.clamps.curve import LineClamp
from classy_blocks.modify.clamps.free import FreeClamp
from classy_blocks.modify.find.geometric import GeometricFinder
from classy_blocks.modify.optimizer import Optimizer


class ConcurrentOptimizerTests(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh()

        grid = Grid([0, 0, 0], [3, 3, 0], 3, 3)
        stack = ExtrudedStack(grid, 3, 3)

        self.mesh.add(stack)
        self.mesh.assemble()

        self.optimizer = Optimizer(self.mesh)
        self.finder = GeometricFinder(self.mesh)

    def get_vertex(self, position):
        return list(self.finder.find_in_sphere(position))[0]

    def test_slopes(self):
        free_vertex = self.get_vertex([1, 1, 1])
        free_clamp = FreeClamp(free_vertex)
        self.optimizer.release_vertex(free_clamp)

        free_vertex.translate([0.2, -0.2, 0])

        line_vertex = self.get_vertex([1, 0, 0])
        line_clamp = LineClamp(line_vertex, [0, 0, 0], [3, 0, 0])
        self.optimizer.release_vertex(line_clamp)

        line_vertex.translate([-0.5, 0, 0])

        self.optimizer.grid.get_slopes()
