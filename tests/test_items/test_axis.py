from parameterized import parameterized

from classy_blocks.items.block import Block
from tests.fixtures.block import BlockTestCase


class AxisTests(BlockTestCase):
    """Tests of the Axis object"""

    def add_blocks(self) -> tuple[Block, Block]:
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)
        block_1.add_neighbour(block_0)

        return block_0, block_1

    def test_length_avg(self):
        """Average length"""
        # all edges are straight and of length 1,
        # except one that has a curved edge
        block = self.make_block(0)
        length = block.axes[0].wires.length
        self.assertAlmostEqual(length, 1.0397797556255037)

    def test_is_aligned_exception(self):
        """Raise an exception when axes are not aligned"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.axes[0].add_neighbour(block_1.axes[0])

        with self.assertRaises(RuntimeError):
            _ = block_0.axes[0].is_aligned(block_1.axes[0])

    @parameterized.expand(((1,), (2,)))
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

    def test_sequential_before(self):
        block_0, block_1 = self.add_blocks()

        calculated_before = set(before.wire for before in block_1.wires[0][1].before)

        expected_before = {block_0.wires[0][1]}

        self.assertSetEqual(calculated_before, expected_before)

    def test_sequential_after(self):
        block_0, block_1 = self.add_blocks()

        calculated_after = set(after.wire for after in block_0.wires[0][1].after)
        expected_after = {block_1.wires[0][1]}

        self.assertSetEqual(calculated_after, expected_after)

    def test_is_inline(self):
        block_0, block_1 = self.add_blocks()

        self.assertTrue(block_0.axes[0].is_inline(block_1.axes[0]))

    def test_is_not_inline(self):
        block_0 = self.make_block(0)
        block_2 = self.make_block(2)
        block_0.add_neighbour(block_2)
        block_2.add_neighbour(block_0)

        self.assertFalse(block_0.axes[0].is_inline(block_2.axes[0]))
