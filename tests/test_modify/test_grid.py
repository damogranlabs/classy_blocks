import numpy as np
from parameterized import parameterized

from classy_blocks.construct.flat.sketches.grid import Grid as GridSketch
from classy_blocks.construct.stack import ExtrudedStack
from classy_blocks.mesh import Mesh
from classy_blocks.optimize.grid import Grid
from classy_blocks.util import functions as f
from tests.fixtures.mesh import MeshTestCase


class GridTests(MeshTestCase):
    def get_grid(self, mesh: Mesh) -> Grid:
        points = np.array([vertex.position for vertex in mesh.vertices])
        addresses = [block.indexes for block in mesh.blocks]

        return Grid(points, addresses)

    def setUp(self):
        super().setUp()
        self.grid = self.get_grid(self.mesh)

    def test_cells_quantity(self):
        self.assertEqual(len(self.grid.cells), len(self.mesh.blocks))

    def test_junctions_quantity(self):
        self.assertEqual(len(self.grid.junctions), len(self.mesh.vertices))

    def test_junction_boundary(self):
        # In this case, ALL junctions are on boundary
        for junction in self.grid.junctions:
            self.assertTrue(junction.is_boundary)

    def test_junction_internal(self):
        sketch = GridSketch([0, 0, 0], [1, 1, 0], 2, 2)
        stack = ExtrudedStack(sketch, 1, 2)

        mesh = Mesh()
        mesh.add(stack)

        try:
            mesh.assemble()  # will fail because there are no chops
        except:
            pass

        grid = self.get_grid(mesh)

        for junction in grid.junctions:
            if f.norm(junction.point - f.vector(0.5, 0.5, 0.5)) < 0.01:
                self.assertFalse(junction.is_boundary)
                continue

            self.assertTrue(junction.is_boundary)

    @parameterized.expand(((0, 1), (1, 2), (2, 3), (3, 1), (4, 1), (5, 2), (6, 3), (7, 1)))
    def test_junction_cells(self, index, count):
        """Each junction contains cells that include that vertex"""
        self.assertEqual(len(self.grid.junctions[index].cells), count)

    @parameterized.expand(((0, "right", 1), (1, "left", 0), (1, "back", 2), (2, "front", 1)))
    def test_cell_neighbours(self, parent, orient, neighbour):
        self.assertEqual(self.grid.cells[parent].neighbours[orient], self.grid.cells[neighbour])

    @parameterized.expand(((0, 3), (1, 4), (2, 5), (3, 3)))
    def test_connections(self, junction, count):
        self.assertEqual(len(self.grid.junctions[junction].connections), count)
