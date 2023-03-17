from typing import List
from parameterized import parameterized

from tests.fixtures.data import DataTestCase

from classy_blocks.construct.edges import Arc, PolyLine, Spline
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
        face_edges = [Arc([0.5, 0.5, 0])]

        edges = self.el.add_from_operation(vertices, face_edges, 'bottom')

        self.assertEqual(len(edges), 1)
        self.assertEqual(len(self.el.edges), 1)
    
    def test_add_existing(self):
        """Add the same edges twice"""
        vertices = self.get_vertices(0)
        face_edges = [Arc([0.5, 0.5, 0])]

        self.el.add_from_operation(vertices, face_edges, 'bottom')
        self.el.add_from_operation(vertices, face_edges, 'bottom')

        self.assertEqual(len(self.el.edges), 1)

    def test_add_invalid(self):
        """Add a circular edge that's actually a line"""
        vertices = self.get_vertices(0)
        face_edges = [Arc([0.5, 0, 0])]

        self.el.add_from_operation(vertices, face_edges, 'bottom')

        self.assertEqual(len(self.el.edges), 0)

    def test_vertex_order(self):
        """Assure vertices maintain consistent order"""
        vertices = self.get_vertices(0)

        # add some custom edges
        bottom_face_edges = [None, None, Spline([[0.7, 1.3, 0], [0.3, 1.3, 0]]), None]
        top_face_edges = [None, None, PolyLine([[0.7, 1.1, 1], [0.3, 1.1, 1]]), None]
        
        self.el.add_from_operation(vertices, bottom_face_edges, 'bottom')
        self.el.add_from_operation(vertices, top_face_edges, 'top')
        
        self.assertEqual(len(self.el.edges), 2)

        self.assertEqual(self.el.edges[0].vertex_1.index, 2)
        self.assertEqual(self.el.edges[0].vertex_2.index, 3)

        self.assertEqual(self.el.edges[1].vertex_1.index, 6)
        self.assertEqual(self.el.edges[1].vertex_2.index, 7)