from parameterized import parameterized
from tests import fixtures

from classy_blocks.items.vertex import Vertex
from classy_blocks.items.block import Block

class BlockTests(fixtures.FixturedTestCase):
    """Block item tests"""
    def make_block(self, index:int) -> Block:
        """The test subject"""
        data = self.get_single_data(index)
        indexes = self.get_vertex_indexes(index)
        vertices = [Vertex(fixtures.fl[i], i) for i in indexes] + \
            [Vertex(fixtures.cl[i], i+len(fixtures.fl)) for i in indexes]

        return Block(data, index, vertices, [])

    @parameterized.expand((
        ((1, 2), (0, 3)), # corner_1, corner_2 of block_0 and corner_1, corner_2 of block_1
        ((5, 6), (4, 7)),
        ((1, 5), (0, 4)),
        ((2, 6), (3, 7)),
    ))
    def test_add_neighbour_1_wires(self, this_corners, nei_corners):
        """Two blocks that share a 'side' a.k.a. face"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)

        # these two blocks share the whole face (1 2 6 5)
        # 4 wires altogether
        self.assertEqual(
            block_0.frame[this_corners[0]][this_corners[1]].coincidents,
            {block_1.frame[nei_corners[0]][nei_corners[1]]}
        )

    def test_add_neighbour_1_axes(self):
        """Two blocks that share a 'side' a.k.a. face"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)

        # block_1 is block_0's neighbour in axes 1 and 2
        self.assertListEqual(block_0.frame.axes[0].neighbours, [])
        self.assertListEqual(block_0.frame.axes[1].neighbours, [block_1.frame.axes[1]])
        self.assertListEqual(block_0.frame.axes[2].neighbours, [block_1.frame.axes[2]])

    def test_add_neighbour_2_wires(self):
        """Two blocks that share an edge only"""
        block_0 = self.make_block(0)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_2)

        # there must be only 1 wire that only has 1 neighbour
        self.assertEqual(
            block_0.frame[2][6].coincidents,
            {block_2.frame[0][4]}
        )

    def test_add_neighbour_2_axes(self):
        """Two blocks that share an edge only"""
        block_0 = self.make_block(0)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_2)
        # block_2 is block_0'2 neighbour only on axis 2
        self.assertListEqual(block_0.frame.axes[0].neighbours, [])
        self.assertListEqual(block_0.frame.axes[1].neighbours, [])
        self.assertListEqual(block_0.frame.axes[2].neighbours, [block_2.frame.axes[2]])


    def test_add_neighbour_3(self):
        """Where three blocks meet, there's a wire with 2 coincident wires"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_1)
        block_0.add_neighbour(block_2)

        self.assertEqual(
            block_0.frame[2][6].coincidents,
            {block_1.frame[3][7], block_2.frame[0][4]}
        )
    
    def test_add_neighbour_twice(self):
        """Add the same neighbour twice"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)
        block_0.add_neighbour(block_1)

        self.assertEqual(len(block_0.frame.axes[1].neighbours), 1)
    
    def test_add_self(self):
        """Add the same block as a neighbour"""
        block_0 = self.make_block(0)

        block_0.add_neighbour(block_0)

        self.assertEqual(len(block_0.frame.axes[0].neighbours), 0)