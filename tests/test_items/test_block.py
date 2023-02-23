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

    def test_add_neighbour_1(self):
        """Two blocks that share a 'side' a.k.a. face"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)

        block_0.add_neighbour(block_1)

        # these two blocks share the whole face; there must be 4 wires that each
        # contains a references to a coincident wire
        n_wires_0 = 0
        n_wires_1 = 0

        for wire in block_0.frame.wires:
            n_wires_0 += len(wire.coincidents)

        for wire in block_1.frame.wires:
            n_wires_1 += len(wire.coincidents)
        
        self.assertEqual(n_wires_0, 4)
        self.assertEqual(n_wires_1, 4)
    
    def test_add_neighbour_2(self):
        """Two blocks that share an edge only"""
        block_0 = self.make_block(0)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_2)

        # there must be only 1 wire that only has 1 neighbour
        n_wires_0 = 0
        n_wires_2 = 0

        for wire in block_0.frame.wires:
            n_wires_0 += len(wire.coincidents)
        
        for wire in block_2.frame.wires:
            n_wires_2 += len(wire.coincidents)
        
        self.assertEqual(n_wires_0, 1)
        self.assertEqual(n_wires_2, 1)

    def test_add_neighbour_3(self):
        """Where three blocks meet, there's a wire with 2 coincident wires"""
        block_0 = self.make_block(0)
        block_1 = self.make_block(1)
        block_2 = self.make_block(2)

        block_0.add_neighbour(block_1)
        block_0.add_neighbour(block_2)

        #self.assertEqual(block_0.frame[2][6].
    
    def test_add_neighbour_twice(self):
        """Add the same neighbour twice"""
    
    def test_add_self(self):
        """Add the same block as a neighbour"""