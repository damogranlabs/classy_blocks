import unittest

from parameterized import parameterized

from classy_blocks.grading.autograding.inflation.params import InflationParams
from classy_blocks.grading.autograding.inflation.rules import InflationRules
from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.wire import Wire


class InflationRulesTests(unittest.TestCase):
    def setUp(self):
        params = InflationParams(1e-3, 0.1, 1.2, 30, 2)
        self.rules = InflationRules(params)

    def get_info(self, length, starts_at_wall, ends_at_wall) -> WireInfo:
        wire = Wire([Vertex([0, 0, 0], 0), Vertex([length, 0, 0], 1)], 0, 0, 1)
        return WireInfo(wire, starts_at_wall, ends_at_wall)

    def test_get_count_bulk(self):
        count = self.rules.get_count(1, False, False)

        self.assertEqual(count, 10)

    @parameterized.expand(
        (
            (1, 25),
            (2, 35),
            (3, 45),
            (0.1, 17),
            (0.05, 17),
            (0.01, 7),
            (0.002, 2),
            (0.001, 1),
            (0.0005, 1),
        )
    )
    def test_get_count_wall(self, length, count):
        # count calculation when one vertex of the wire is at wall;
        # numbers checked with manual spreadsheet calculation
        self.assertEqual(self.rules.get_count(length, True, False), count)

    @parameterized.expand(
        (
            (1, 40),  # 0
            (2, 50),  # 1
            (3, 60),  # 2
            (0.1, 34),  # 3
            (0.05, 22),  # 4
            (0.01, 10),  # 5
            (0.002, 2),  # 6
            (0.001, 2),  # 7
            (0.0005, 2),  # 8
        )
    )
    def test_get_count_double_wall(self, length, count):
        # count when both vertices are at wall
        # numbers checked with manual spreadsheet calculation
        half_count = self.rules.get_count(length / 2, True, False)

        self.assertEqual(2 * half_count, count)

    @parameterized.expand(
        (
            # not squeezed - enough room, low cell count
            (1, 24, False),  # 0
            (2, 34, False),  # 1
            (3, 44, False),  # 2
            (0.3, 16, False),  # 3
            # squeezed: enough room, high cell count
            (0.1, 25, True),  # 4
            (1, 35, True),  # 5
            (2, 45, True),  # 6
            # squeezed: not enough room, cell count doesn't matter
            (0.09, 2, True),  # 7
        )
    )
    def test_is_squeezed_wall(self, length, count, squeezed):
        info = self.get_info(length, True, False)

        self.assertEqual(self.rules.is_squeezed(count, info), squeezed)

    @parameterized.expand(
        (
            (0.01, 9, False),
            (0.01, 0, False),
            (0.01, 100, True),
        )
    )
    def test_is_squeezed_bulk(self, length, count, squeezed):
        # a test of SmoothGrader, actually
        info = self.get_info(length, False, False)

        self.assertEqual(self.rules.is_squeezed(count, info), squeezed)
