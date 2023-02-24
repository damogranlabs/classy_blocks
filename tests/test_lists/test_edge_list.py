from tests.fixtures.data import DataTestCase

from classy_blocks.lists.vertex_list import VertexList
from classy_blocks.lists.edge_list import EdgeList


class EdgeListTests(DataTestCase):
    def setUp(self):
        self.blocks = DataTestCase.get_all_data()
        self.vl = VertexList()
        self.el = EdgeList()

    def test_add_new(self):
        """Add an edge when no such thing exists"""
        vertices = self.vl.add(self.blocks[0].points)
        edges = self.el.add(self.blocks[0], vertices)

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