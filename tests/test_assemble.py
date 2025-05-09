import unittest

from parameterized import parameterized

from classy_blocks.assemble.connection_registry import HexConnectionRegistry, get_key
from classy_blocks.assemble.point_registry import HexPointRegistry
from classy_blocks.construct.shapes.cylinder import Cylinder


class RegistryTestBase(unittest.TestCase):
    def get_hex_point_registry(self) -> HexPointRegistry:
        shape = Cylinder([0, 0, 0], [0, 0, 1], [1, 0, 0])
        registry = HexPointRegistry.from_operations(shape.operations)

        return registry


class PointRegistryTests(RegistryTestBase):
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


class ConnectionRegistryTests(RegistryTestBase):
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
