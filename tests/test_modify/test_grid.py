import numpy as np
from parameterized import parameterized

from classy_blocks.optimize.grid import Grid
from tests.fixtures.mesh import MeshTestCase


class GridTests(MeshTestCase):
    def setUp(self):
        super().setUp()

        points = np.array([vertex.position for vertex in self.mesh.vertices])
        addresses = [block.indexes for block in self.mesh.blocks]
        self.grid = Grid(points, addresses)

    def test_cells_quantity(self):
        self.assertEqual(len(self.grid.cells), len(self.mesh.blocks))

    def test_junctions_quantity(self):
        self.assertEqual(len(self.grid.junctions), len(self.mesh.vertices))

    @parameterized.expand(((0, 1), (1, 2), (2, 3), (3, 1), (4, 1), (5, 2), (6, 3), (7, 1)))
    def test_junction_cells(self, index, count):
        """Each junction contains cells that include that vertex"""
        self.assertEqual(len(self.grid.junctions[index].cells), count)

    @parameterized.expand(((0, "right", 1), (1, "left", 0), (1, "back", 2), (2, "front", 1)))
    def test_cell_neighbours(self, parent, orient, neighbour):
        self.assertEqual(self.grid.cells[parent].neighbours[orient], self.grid.cells[neighbour])
