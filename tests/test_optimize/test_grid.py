import numpy as np
from parameterized import parameterized

from classy_blocks.construct.flat.sketches.disk import OneCoreDisk
from classy_blocks.construct.flat.sketches.grid import Grid as GridSketch
from classy_blocks.construct.stack import ExtrudedStack
from classy_blocks.mesh import Mesh
from classy_blocks.optimize.grid import HexGrid, QuadGrid
from classy_blocks.util import functions as f
from tests.fixtures.mesh import MeshTestCase
from tests.test_optimize.optimize_fixtures import SketchTestsBase


class HexGridTests(MeshTestCase):
    def get_grid(self, mesh: Mesh) -> HexGrid:
        points = np.array([vertex.position for vertex in mesh.vertices])
        addresses = [block.indexes for block in mesh.blocks]

        return HexGrid(points, addresses)

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

        mesh.assemble()

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
    def test_neighbours(self, junction, count):
        self.assertEqual(len(self.grid.junctions[junction].neighbours), count)


class QuadGridTests(SketchTestsBase):
    def test_from_sketch(self):
        sketch = OneCoreDisk([0, 0, 0], [1, 0, 0], [0, 0, 1])
        grid = QuadGrid.from_sketch(sketch)

        self.assertEqual(len(grid.cells), 5)
        self.assertEqual(len(grid.junctions), 8)

    @parameterized.expand(
        (
            (0, 4),
            (1, 3),
            (2, 3),
            (3, 3),
            (4, 3),
        )
    )
    def test_neighbour_cells(self, i_cell, n_neighbours):
        sketch = OneCoreDisk([0, 0, 0], [1, 0, 0], [0, 0, 1])
        grid = QuadGrid.from_sketch(sketch)

        neighbour_count = 0
        for n in grid.cells[i_cell].neighbours.values():
            if n is not None:
                neighbour_count += 1

        self.assertEqual(neighbour_count, n_neighbours)

    def test_positions(self):
        np.testing.assert_equal(self.grid.points, self.positions)

    @parameterized.expand(
        (
            (0, {8, 1}),
            (1, {0, 2, 6}),
            (2, {1, 3, 7}),
            (3, {2, 4}),
            (4, {3, 5, 7}),
            (5, {4, 6}),
            (6, {8, 1, 5, 7}),
            (7, {2, 4, 6}),
            (8, {0, 6}),
        )
    )
    def test_find_neighbours(self, i_junction, expected_neighbours):
        # A random blocking (quadding)
        positions = np.zeros((9, 3))
        indexes = [[1, 2, 7, 6], [2, 3, 4, 7], [7, 4, 5, 6], [0, 1, 6, 8]]

        grid = QuadGrid(positions, indexes)

        self.assertSetEqual(expected_neighbours, {junction.index for junction in grid.junctions[i_junction].neighbours})

    def test_fixed_points(self):
        # Monocylinder, core is quads[0]
        positions = np.zeros((8, 3))
        indexes = [
            [0, 1, 2, 3],
            [0, 4, 5, 1],
            [1, 5, 6, 2],
            [2, 6, 7, 3],
            [3, 7, 4, 0],
        ]

        grid = QuadGrid(positions, indexes)
        fixed_points = set()

        for cell in grid.cells:
            fixed_points.update(cell.boundary)

        self.assertSetEqual(
            fixed_points,
            {4, 5, 6, 7},
        )
