import unittest

from tests.fixtures import block_data, FixturedTestCase

class BlockTests(unittest.TestCase):
    def setUp(self):
        self.blocks = FixturedTestCase.get_blocks()

    def test_create(self):
        self.assertEqual(len(self.blocks), 3)

    def test_edges(self):
        for i, data in enumerate(block_data):
            self.assertEqual(len(data.edges), len(self.blocks[i].edges))
    
    def test_chops(self):
        for i, data in enumerate(block_data):
            self.assertEqual(len(data.counts), len(self.blocks[i].chops))

    def test_patches(self):
        for i, data in enumerate(block_data):
            for patch in data.patches:
                orients = patch[0]
                name = patch[1]

                if isinstance(orients, str):
                    orients = [orients]

                for orient in orients:
                    self.assertEqual(self.blocks[i].sides[orient].patch, name)

    def test_description(self):
        for i, data in enumerate(block_data):
            self.assertEqual(self.blocks[i].description, data.description)
            self.assertEqual(self.blocks[i].cell_zone, data.cell_zone)