import unittest

from classy_blocks.grading.autograding.fixed.rules import FixedCountRules
from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.autograding.simple.rules import SimpleRules
from classy_blocks.grading.autograding.smooth.rules import SmoothRules
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.wire import Wire


class GradingRulesTestsBase(unittest.TestCase):
    def get_info(self, length: float = 1, starts_at_wall: bool = False, ends_at_wall: bool = False) -> WireInfo:
        vertices = [Vertex([0, 0, 0], 0), Vertex([length, 0, 0], 1)]
        wire = Wire(vertices, 0, 0, 1)
        info = WireInfo(wire, starts_at_wall, ends_at_wall)

        return info


class FixedGradingRulesTests(GradingRulesTestsBase):
    count = 10

    def setUp(self):
        self.rules = FixedCountRules(self.count)

    def test_get_count(self):
        self.assertEqual(self.rules.get_count(100, True, True), self.count)

    def test_is_squeezed(self):
        self.assertTrue(self.rules.is_squeezed(self.count, self.get_info()))

    def test_get_chops(self):
        with self.assertRaises(RuntimeError):
            self.rules.get_chops(100, self.get_info())


class SimpleGradingRulesTests(GradingRulesTestsBase):
    size = 0.05

    def setUp(self):
        self.rules = SimpleRules(self.size)

    def test_get_count(self):
        self.assertEqual(self.rules.get_count(1, False, False), 1 / 0.05)

    def test_is_squeezed(self):
        self.assertTrue(self.rules.is_squeezed(100, self.get_info()))

    def test_get_chops(self):
        chops = self.rules.get_chops(10, self.get_info())

        self.assertEqual(len(chops), 1)
        self.assertEqual(chops[0].count, 10)


class SmoothGradingRulesTests(GradingRulesTestsBase):
    size = 0.05

    def setUp(self):
        self.rules = SmoothRules(self.size)

    def test_get_count_zero(self):
        self.assertEqual(self.rules.get_count(self.size / 10, False, False), 2)

    def test_get_count(self):
        self.assertEqual(self.rules.get_count(1, False, False), 1 / self.size)

    def test_squeezed(self):
        self.assertTrue(self.rules.is_squeezed(100, self.get_info()))

    def test_not_squeezed(self):
        self.assertFalse(self.rules.is_squeezed(1, self.get_info()))

    def test_chops_squeezed(self):
        chops = self.rules.get_chops(100, self.get_info(0.1))

        self.assertEqual(len(chops), 1)
        self.assertEqual(chops[0].count, 100)

    def test_chops_not_squeezed(self):
        chops = self.rules.get_chops(10, self.get_info())

        self.assertEqual(len(chops), 2)
        self.assertEqual(sum(chop.count for chop in chops), 10)
