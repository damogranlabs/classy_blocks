from parameterized import parameterized

from classy_blocks.items.block import Block

from tests.fixtures.block import BlockTestCase

class AxisTests(BlockTestCase):
    """Tests of the Axis object"""
    def test_lengths(self):
        """Block dimensions"""
        # Not all wires on this axis are of the same length
        lengths = self.make_block(0).axes[0].lengths

        self.assertEqual(len(lengths), 4)

        # all edges in this axis are straight but 0-1
        self.assertNotEqual(lengths[0], lengths[1])

        self.assertEqual(lengths[1], lengths[2])
        self.assertEqual(lengths[2], lengths[3])

    def test_length_default(self):
        """Use average length when no chops are defined"""
        self.assertEqual(self.make_block(0).axes[0].length, 1.0397797556255037 )

    def test_length_min(self):
        """Minimum length"""
        block = self.make_block(0)
        block.axes[0].chops[0].take = 'min'
        self.assertEqual(block.axes[0].length, 1)

    def test_length_max(self):
        """Maximum length"""
        block = self.make_block(0)
        block.axes[0].chops[0].take = 'max'
        self.assertEqual(block.axes[0].length, 1.1591190225020154)

    def test_length_avg(self):
        """Average length"""
        block = self.make_block(0)
        block.axes[0].chops[0].take = 'avg'
        self.assertEqual(block.axes[0].length, 1.0397797556255037 )

    def test_is_aligned_exception(self):
        """Raise an exception when axes are not aligned"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.axes[0].add_neighbour(block_1.axes[0])

        with self.assertRaises(RuntimeError):
            _ = block_0.axes[0].is_aligned(block_1.axes[0])

    @parameterized.expand(((1, ), (2, )))
    def test_is_aligned(self, axis):
        """Returns True when axes are aligned"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.axes[axis].add_neighbour(block_1.axes[axis])

        self.assertTrue(block_0.axes[axis].is_aligned(block_1.axes[axis]))

    def test_is_not_aligned(self):
        """Return False when axes are not aligned"""
        block_0 = self.make_block(0)

        # turn block_1 upside-down
        vertices_1 = self.make_vertices(1)
        vertices_1 = [vertices_1[i] for i in [7, 6, 5, 4, 3, 2, 1, 0]]

        block_1 = Block(1, vertices_1)

        block_0.add_neighbour(block_1)

        self.assertFalse(block_1.axes[1].is_aligned(block_0.axes[1]))

    def test_grading_self(self):
        """Grading, defined on this axis"""
        block_0 = self.make_block(0)
        
        self.assertEqual(block_0.axes[0].grading.count, 6)
