from tests.fixtures import data
from tests.fixtures.data import DataTestCase

from classy_blocks.items.vertex import Vertex
from classy_blocks.items.block import Block
from classy_blocks.items.edges.factory import factory

class BlockTestCase(DataTestCase):
    """Block item tests"""
    def make_block(self, index:int) -> Block:
        """The test subject"""
        block_data = self.get_single_data(index)
        indexes = self.get_vertex_indexes(index)
        vertices = [Vertex(data.fl[i], i) for i in indexes] + \
            [Vertex(data.cl[i], i+len(data.fl)) for i in indexes]
        
        edges = []

        for edge_data in block_data.edges:
            args = [
                vertices[edge_data.corner_1], # vertex_1
                vertices[edge_data.corner_2], # vertex_2
                edge_data.kind,
                *edge_data.args
            ]
            edges.append(factory.create(*args))

        return Block(block_data, index, vertices, edges)
