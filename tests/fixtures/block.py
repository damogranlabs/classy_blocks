from typing import List, get_args

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.items.block import Block
from classy_blocks.items.edges.factory import factory
from classy_blocks.items.vertex import Vertex
from classy_blocks.types import DirectionType
from tests.fixtures.data import DataTestCase


class BlockTestCase(DataTestCase):
    """Block item tests"""

    def make_vertices(self, index: int) -> List[Vertex]:
        """Generates Vertex objects for testing"""
        data = self.get_single_data(index)
        points = data.points
        indexes = data.indexes

        return [Vertex(p, indexes[i]) for i, p in enumerate(points)]

    def make_block(self, index: int) -> Block:
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

        for i in get_args(DirectionType):
            block.add_chops(i, block_data.chops[i])

        return block

    def make_loft(self, index: int) -> Loft:
        """Creates a Loft for tests that require an operation"""
        vertices = self.make_vertices(index)
        face_1 = Face([vertices[i].position for i in (0, 1, 2, 3)])
        face_2 = Face([vertices[i].position for i in (4, 5, 6, 7)])

        return Loft(face_1, face_2)
