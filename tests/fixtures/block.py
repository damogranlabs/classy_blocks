import dataclasses
from tests.fixtures.data import DataTestCase

from classy_blocks.items.vertex import Vertex
from classy_blocks.items.block import Block
from classy_blocks.items.edges.factory import factory

class BlockTestCase(DataTestCase):
    """Block item tests"""
    def make_block(self, index:int) -> Block:
        """The test subject"""
        block_data = self.get_single_data(index)

        block = Block(block_data.points)
        block.index = index

        for edge_data in block_data.edges:
            block.add_edge(*edge_data)
        
        for axis in (0, 1, 2):
            for chop in block_data.chops[axis]:
                block.chop(axis, **dataclasses.asdict(chop))

        return block
