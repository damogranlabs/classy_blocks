import unittest

from parameterized import parameterized

from classy_blocks.grading.autograding.params.base import WireInfo
from classy_blocks.grading.autograding.params.inflation import InflationGraderParams
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.wire import Wire


class InflationParamsTests(unittest.TestCase):
    def setUp(self):
        self.params = InflationGraderParams(1e-3, 0.1, 1.2, 30, 2)

    def get_info(self, length, starts_at_wall, ends_at_wall) -> WireInfo:
        wire = Wire([Vertex([0, 0, 0], 0), Vertex([length, 0, 0], 1)], 0, 0, 1)
        return WireInfo(wire, starts_at_wall, ends_at_wall)

    def test_get_count_bulk(self):
        count = self.params.get_count(1, False, False)

        self.assertEqual(count, 10)

    @parameterized.expand(
        (
            (1, 24),
            (2, 34),
            (3, 44),
            (0.1, 16),
            (0.05, 14),
            (0.01, 7),
            (0.002, 2),
            (0.001, 2),
            (0.0005, 1),
        )
    )
    def test_get_count_wall(self, length, count):
        # count calculation when one vertex of the wire is at wall;
        # numbers checked with manual spreadsheet calculation
        self.assertEqual(self.params.get_count(length, True, False), count)

    @parameterized.expand(
        (
            (1, 38),  # 0
            (2, 48),  # 1
            (3, 58),  # 2
            (0.1, 28),  # 3
            (0.05, 20),  # 4
            (0.01, 8),  # 5
            (0.002, 4),  # 6
            (0.001, 2),  # 7
            (0.0005, 2),  # 8
        )
    )
    def test_get_count_double_wall(self, length, count):
        # count when both vertices are at wall
        # numbers checked with manual spreadsheet calculation
        half_count = self.params.get_count(length / 2, True, False)

        self.assertEqual(2 * half_count, count)

    @parameterized.expand(
        (
            # not squeezed - enough room, low cell count
            (1, 24, False),  # 0
            (2, 34, False),  # 1
            (3, 44, False),  # 2
            (0.3, 16, False),  # 3
            # squeezed: enough room, high cell count
            (1, 25, True),  # 4
            (2, 35, True),  # 5
            (3, 45, True),  # 6
            # squeezed: not enough room, cell count doesn't matter
            (0.2, 16, True),  # 7
            (0.2, 0, True),  # 8
        )
    )
    def test_is_squeezed_wall(self, length, count, squeezed):
        info = self.get_info(length, True, False)

        self.assertEqual(self.params.is_squeezed(count, info), squeezed)

    @parameterized.expand(
        (
            (1, 9, False),
            (1, 0, False),
            (1, 11, True),
        )
    )
    def test_is_squeezed_bulk(self, length, count, squeezed):
        # a test of SmoothGrader, actually
        info = self.get_info(length, False, False)

        self.assertEqual(self.params.is_squeezed(count, info), squeezed)
