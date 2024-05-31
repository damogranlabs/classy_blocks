import numpy as np
from parameterized import parameterized

from classy_blocks.modify.cell import Cell, NoCommonSidesError
from tests.fixtures.mesh import MeshTestCase


class CellTests(MeshTestCase):
    def setUp(self):
        super().setUp()

    @property
    def mesh_points(self):
        return np.array([vertex.position for vertex in self.mesh.vertices])

    @parameterized.expand(
        (
            (0, 1, 4),
            (0, 2, 2),
            (1, 0, 4),
            (1, 2, 4),
        )
    )
    def test_common_vertices(self, index_1, index_2, count):
        block_1 = self.mesh.blocks[index_1]
        block_2 = self.mesh.blocks[index_2]

        cell_1 = Cell(block_1, self.mesh_points)
        cell_2 = Cell(block_2, self.mesh_points)

        self.assertEqual(len(cell_1.get_common_vertices(cell_2)), count)

    @parameterized.expand(((0, 0, 0), (0, 1, 1), (1, 1, 0), (1, 8, 1)))
    def test_get_corner(self, block, vertex, corner):
        cell = Cell(self.mesh.blocks[block], self.mesh_points)

        self.assertEqual(cell.get_corner(vertex), corner)

    @parameterized.expand(((0, 1, "right"), (1, 0, "left"), (1, 2, "back")))
    def test_get_common_side(self, index_1, index_2, orient):
        cell_1 = Cell(self.mesh.blocks[index_1], self.mesh_points)
        cell_2 = Cell(self.mesh.blocks[index_2], self.mesh_points)

        self.assertEqual(cell_1.get_common_side(cell_2), orient)

    def test_no_common_sides(self):
        with self.assertRaises(NoCommonSidesError):
            cell_1 = Cell(self.mesh.blocks[0], self.mesh_points)
            cell_2 = Cell(self.mesh.blocks[2], self.mesh_points)

            cell_1.get_common_side(cell_2)

    def test_quality_good(self):
        cell = Cell(self.mesh.blocks[0], self.mesh_points)

        self.assertLess(cell.quality, 1)

    def test_quality_bad(self):
        block = self.mesh.blocks[0]
        block.vertices[0].move_to([-10, -10, -10])

        cell = Cell(block, self.mesh_points)

        self.assertGreater(cell.quality, 100)
