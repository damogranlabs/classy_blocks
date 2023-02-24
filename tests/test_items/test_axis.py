from parameterized import parameterized

from tests.fixtures.block import BlockTestCase

class AxisTests(BlockTestCase):

    def test_lengths(self):
        """Block dimensions"""
        # Not all wires on this axis are of the same length
        lengths = self.make_block(0).frame.axes[0].lengths

        self.assertEqual(len(lengths), 4)

        # all edges in this axis are straight but 0-1
        self.assertNotEqual(lengths[0], lengths[1])

        self.assertEqual(lengths[1], lengths[2])
        self.assertEqual(lengths[2], lengths[3])

    def test_length_default(self):
        """Use average length when no chops are defined"""
        self.assertEqual(self.make_block(0).frame.axes[0].length, 1.0397797556255037 )

    def test_length_min(self):
        """Minimum length"""
        block = self.make_block(0)
        block.frame.axes[0].chops[0].take = 'min'
        self.assertEqual(block.frame.axes[0].length, 1)

    def test_length_max(self):
        """Maximum length"""
        block = self.make_block(0)
        block.frame.axes[0].chops[0].take = 'max'
        self.assertEqual(block.frame.axes[0].length, 1.1591190225020154)

    def test_length_avg(self):
        """Average length"""
        block = self.make_block(0)
        block.frame.axes[0].chops[0].take = 'avg'
        self.assertEqual(block.frame.axes[0].length, 1.0397797556255037 )

    def test_is_aligned_exception(self):
        """Raise an exception when axes are not aligned"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.frame.axes[0].add_neighbour(block_1.frame.axes[0])

        with self.assertRaises(RuntimeError):
            _ = block_0.frame.axes[0].is_aligned(block_1.frame.axes[0])

    @parameterized.expand(((1, ), (2, )))
    def test_is_aligned(self, axis):
        """Returns True when axes are aligned"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.frame.axes[axis].add_neighbour(block_1.frame.axes[axis])

        self.assertTrue(block_0.frame.axes[axis].is_aligned(block_1.frame.axes[axis]))

    def test_is_not_aligned(self):
        # TODO
        pass

    def test_grading_self(self):
        """Grading, defined on this axis"""
        block_0 = self.make_block(0)
        
        self.assertEqual(block_0.frame.axes[0].grading.count, 6)
    
    def test_grading_other(self):
        """Get grading, defined on neighbour's axis"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)

        self.assertEqual(block_0.frame.axes[1].grading.count, 6)