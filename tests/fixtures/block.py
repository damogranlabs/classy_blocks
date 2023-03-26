from typing import List

from tests.fixtures.data import DataTestCase

from classy_blocks.items.vertex import Vertex
from classy_blocks.items.block import Block
from classy_blocks.items.edges.factory import factory

class BlockTestCase(DataTestCase):
    """Block item tests"""
    def make_vertices(self, index:int) -> List[Vertex]:
        data = self.get_single_data(index)
        points = data.points
        indexes = data.indexes

        return [Vertex(p, indexes[i]) for i, p in enumerate(points)]

    def make_block(self, index:int) -> Block:
        """The test subject"""
        block_data = self.get_single_data(index)
        vertices = self.make_vertices(index)

        block = Block(index, vertices)

        for edge_data in block_data.edges:
            corner_1 = edge_data[0]
            corner_2 = edge_data[1]
            vertex_1 = vertices[corner_1]
            vertex_2 = vertices[corner_2]
            edge = factory.create(vertex_1, vertex_2, edge_data[2])

            block.add_edge(corner_1, corner_2, edge)

        for axis in (0, 1, 2):
            for chop in block_data.chops[axis]:
                block.chop(axis, chop)

        return block
