from tests.fixtures import FixturedTestCase

from classy_blocks.lists.vertex_list import VertexList
from classy_blocks.lists.edge_list import EdgeList

class EdgeListTests(FixturedTestCase):
    def setUp(self):
        self.blocks = self.get_blocks()
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

