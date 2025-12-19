import unittest

from classy_blocks.grading.define.collector import Chop, ChopCollector


class CollectorTests(unittest.TestCase):
    def test_chop_edge_single(self):
        c = ChopCollector()

        c.chop_edge(0, 1, Chop(count=10))

        edge_chops = 0
        for beam in c.edge_chops.get_all_beams():
            if beam:
                edge_chops += 1
        self.assertEqual(edge_chops, 1)

    def test_chop_edge_multiple(self):
        c = ChopCollector()

        c.chop_edge(0, 1, Chop(count=10))
        c.chop_edge(3, 2, Chop(count=10))

        edge_chops = 0
        for beam in c.edge_chops.get_all_beams():
            if beam:
                edge_chops += 1
        self.assertEqual(edge_chops, 2)

    def test_is_not_edge_chopped(self):
        c = ChopCollector()

        c.chop_axis(0, Chop(count=10))

        self.assertFalse(c.is_edge_chopped)

    def test_is_edge_chopped(self):
        c = ChopCollector()

        c.chop_edge(0, 1, Chop(count=10))

        self.assertTrue(c.is_edge_chopped)
