from classy_blocks.base.exceptions import UndefinedGradingsError
from classy_blocks.lists.block_list import BlockList
from tests.fixtures.block import BlockTestCase


class BlockListTests(BlockTestCase):
    def setUp(self):
        self.bl = BlockList()

    def test_add(self):
        """Add a block and check the results"""
        self.bl.add(self.make_block(0))
        self.assertEqual(self.bl.blocks[0].index, 0)

        self.bl.add(self.make_block(1))
        self.assertEqual(self.bl.blocks[1].index, 1)

    def test_propagate_gradings_ok(self):
        """Define all block's grading data"""
        for index in (0, 1, 2):
            self.bl.add(self.make_block(index))

        self.bl.propagate_gradings()

        for block in self.bl.blocks:
            self.assertTrue(block.is_defined)

    def test_propagate_gradings_exception(self):
        """Raise an exception when there's not enough grading data"""
        blocks = [self.make_block(i) for i in (0, 1, 2)]

        blocks[0].axes[0].chops = []

        for block in blocks:
            self.bl.add(block)

        with self.assertRaises(UndefinedGradingsError):
            self.bl.propagate_gradings()

    def test_description(self):
        """Text output for blockMesh"""
        blocks = [self.make_block(i) for i in (0, 1, 2)]

        for block in blocks:
            self.bl.add(block)

        self.bl.propagate_gradings()

        expected = "blocks\n(\n"
        
        for block in blocks:
            expected += block.description
        
        expected += ");\n\n"

        self.assertEqual(self.bl.description, expected)
