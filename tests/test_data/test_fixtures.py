import unittest

from tests.fixtures.data import test_data, DataTestCase

class BlockDataFixturesTests(unittest.TestCase):
    def setUp(self):
        self.data = DataTestCase.get_all_data()

    def test_create(self):
        self.assertEqual(len(self.data), 3)

    def test_edges(self):
        for i, data in enumerate(test_data):
            self.assertEqual(len(data.edges), len(self.data[i].edges))
    
    def test_chops(self):
        for i, data in enumerate(test_data):
            self.assertEqual(len(data.counts), len(self.data[i].axis_chops))

    def test_get_edge_success(self):
        self.assertEqual(self.data[0].get_edge(0, 1).kind, 'arc')
    
    def test_get_edge_fail(self):
        with self.assertRaises(RuntimeError):
            self.data[0].get_edge(2, 3)

    def test_patches(self):
        for i, data in enumerate(test_data):
            for patch in data.patches:
                orients = patch[0]
                name = patch[1]

                if isinstance(orients, str):
                    orients = [orients]

                for orient in orients:
                    self.assertEqual(self.data[i].sides[orient].patch_name, name)

    def test_description(self):
        for i, data in enumerate(test_data):
            self.assertEqual(self.data[i].comment, data.description)
            self.assertEqual(self.data[i].cell_zone, data.cell_zone)
