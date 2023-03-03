from typing import List
from parameterized import parameterized

from tests.fixtures.data import DataTestCase

from classy_blocks.items.vertex import Vertex
from classy_blocks.lists.vertex_list import VertexList
from classy_blocks.lists.edge_list import EdgeList


class EdgeListTests(DataTestCase):
    def setUp(self):
        self.blocks = DataTestCase.get_all_data()
        self.vl = VertexList()
        self.el = EdgeList()

    def get_vertices(self, index:int) -> List[Vertex]:
        vertices = []

        for point in self.get_single_data(index).points:
            vertices.append(self.vl.add(point))
    
        return vertices


    @parameterized.expand(((0, 1), (1, 2), (2, 3), (3, 0)))
    def test_get_corners_bottom(self, corner_1, corner_2):
        """get_corners for a bottom face"""
        output = (corner_1, corner_2)
        self.assertTupleEqual(EdgeList.get_corners(corner_1, 'bottom'), output)

    @parameterized.expand(((0, 4, 5), (1, 5, 6), (2, 6, 7), (3, 7, 4)))
    def test_get_corners_top(self, index, corner_1, corner_2):
        """get_corners for a bottom face"""
        output = (corner_1, corner_2)
        self.assertTupleEqual(EdgeList.get_corners(index, 'top'), output)

    @parameterized.expand(((0, 4), (1, 5), (2, 6), (3, 7)))
    def test_get_corners_side(self, corner_1, corner_2):
        """get_corners for a side corners"""
        output = (corner_1, corner_2)
        self.assertTupleEqual(EdgeList.get_corners(corner_1, 'side'), output)

    def test_add_new(self):
        """Add an edge when no such thing exists"""
        vertices = self.get_vertices(0)
        self.el.add(vertices, self.blocks[0].edges)
        data = self.blocks[0].edges

        self.assertEqual(len(edges), len(self.blocks[0].edges))
        self.assertEqual(len(self.el.edges), len(self.blocks[0].edges))
    
    def test_add_existing(self):
        """Add the same edges twice"""
        vertices = self.vl.add(self.blocks[0].points)
        edges_first = self.el.add(self.blocks[0], vertices)
        edges_second = self.el.add(self.blocks[0], vertices)

        self.assertEqual(len(edges_first), len(edges_second))

        for i, edge in enumerate(edges_first):
            self.assertEqual(id(edge), id(edges_second[i]))

    def test_add_invalid(self):
        """Add a circular edge that's actually a line"""
        vertices = self.vl.add(self.blocks[1].points)
        edges = self.el.add(self.blocks[1], vertices)

        # there's one duplicated edge in the definition but we
        # didn't add block 1
        self.assertEqual(len(edges), 1)

    def test_add_duplicate(self):
        """Do not add a duplicate edge"""
        vertices = self.vl.add(self.blocks[0].points)
        self.el.add(self.blocks[0], vertices)

        # this block has one duplicate and one invalid edge
        vertices = self.vl.add(self.blocks[1].points)
        self.el.add(self.blocks[1], vertices)

        self.assertEqual(len(self.el.edges), len(self.blocks[0].edges))

    def test_vertex_order(self):
        """Assure vertices maintain consistent order"""
        # (as provided by the user, that is)
        # the most low-level way of creating a block is from 'raw' points
        block = self.blocks[0]

        # add some custom edges
        block.edges = []
        block.add_edge(2, 3, 'spline', [[0.7, 1.3, 0], [0.3, 1.3, 0]]) # spline edge
        block.add_edge(6, 7, 'polyLine', [[0.7, 1.1, 1], [0.3, 1.1, 1]]) # weird edge

        vertices = self.vl.add(block.points)
        self.el.add(block, vertices)

        self.assertEqual(len(self.el.edges), 2)

        self.assertEqual(self.el.edges[0].vertex_1.index, 2)
        self.assertEqual(self.el.edges[0].vertex_2.index, 3)

        self.assertEqual(self.el.edges[1].vertex_1.index, 6)
        self.assertEqual(self.el.edges[1].vertex_2.index, 7)