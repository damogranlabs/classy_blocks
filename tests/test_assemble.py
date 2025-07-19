import unittest

from parameterized import parameterized

from classy_blocks.construct.operations.box import Box
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.lookup.cell_registry import CellRegistry
from classy_blocks.lookup.connection_registry import HexConnectionRegistry, get_key
from classy_blocks.lookup.face_registry import HexFaceRegistry
from classy_blocks.lookup.point_registry import HexPointRegistry


class RegistryTests(unittest.TestCase):
    def get_shape(self) -> Cylinder:
        return Cylinder([0, 0, 0], [0, 0, 1], [1, 0, 0])

    def get_hex_point_registry(self) -> HexPointRegistry:
        registry = HexPointRegistry.from_operations(self.get_shape().operations)

        return registry

    @parameterized.expand(
        (
            ([0, 0, 0], 0),
            ([0, 0, 1], 4),
            ([1, 0, 0], 18),
            ([1, 0, 1], 20),
            ([0, 1, 0], 22),
            ([0, 1, 1], 23),
        )
    )
    def test_find_point_index(self, point, index):
        # points are taken from an actual blockMesh'ed cylinder
        registry = self.get_hex_point_registry()

        self.assertEqual(registry.find_point_index(point), index)

    def test_connection(self):
        point_reg = self.get_hex_point_registry()
        conn_reg = HexConnectionRegistry(point_reg.unique_points, point_reg.cell_addressing)

        point_1 = [0, 0, 0]
        point_2 = [0, 0.8, 0]

        index_1 = point_reg.find_point_index(point_1)
        index_2 = point_reg.find_point_index(point_2)

        self.assertTrue(get_key(index_1, index_2) in conn_reg.connections)

    @parameterized.expand(((0, 5), (18, 4)))
    def test_junction(self, point_index, neighbour_count):
        point_reg = self.get_hex_point_registry()
        conn_reg = HexConnectionRegistry(point_reg.unique_points, point_reg.cell_addressing)

        self.assertEqual(len(conn_reg.get_connected_indexes(point_index)), neighbour_count)

    @parameterized.expand(
        (
            (0, "top", {0, 1}),
            (0, "bottom", {0, 2}),
            (1, "bottom", {0, 1}),
            (1, "top", {1}),
            (2, "top", {0, 2}),
            (0, "left", {0}),
            (0, "right", {0}),
            (0, "front", {0}),
            (0, "back", {0}),
        )
    )
    def test_face_registry(self, cell, side, indexes):
        box_ground = Box([0, 0, 0], [1, 1, 1])
        box_up = box_ground.copy().translate([0, 0, 1])
        box_down = box_ground.copy().translate([0, 0, -1])

        preg = HexPointRegistry.from_operations([box_ground, box_up, box_down])
        freg = HexFaceRegistry(preg.cell_addressing)

        self.assertSetEqual(freg.get_cells(cell, side), indexes)

    @parameterized.expand(
        (
            ([0, 0, 0], {0}),
            ([1, 0, 0], {0, 1}),
            ([2, 0, 0], {1}),
            ([1, 1, 0], {0, 1, 2, 3}),
        )
    )
    def test_cell_registry(self, point, cells) -> None:
        boxes: list[Operation] = [
            Box([0, 0, 0], [1, 1, 1]),  # lower left
            Box([0, 0, 0], [1, 1, 1]).translate([1, 0, 0]),  # lower right
            Box([0, 0, 0], [1, 1, 1]).translate([1, 1, 0]),  # upper right
            Box([0, 0, 0], [1, 1, 1]).translate([0, 1, 0]),  # upper left
        ]

        preg = HexPointRegistry.from_operations(boxes)
        creg = CellRegistry(preg.cell_addressing)

        point_index = preg.find_point_index(point)
        self.assertSetEqual(creg.get_near_cells(point_index), cells)
