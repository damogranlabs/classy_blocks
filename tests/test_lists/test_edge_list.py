from typing import List

from tests.fixtures.data import DataTestCase

from classy_blocks.base.exceptions import EdgeNotFoundError
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.edges import Project
from classy_blocks.construct.operations.revolve import Revolve
from classy_blocks.construct.edges import Arc, PolyLine, Spline
from classy_blocks.items.vertex import Vertex
from classy_blocks.lists.vertex_list import VertexList
from classy_blocks.lists.edge_list import EdgeList


class EdgeListTests(DataTestCase):
    def setUp(self):
        self.blocks = DataTestCase.get_all_data()
        self.vl = VertexList()
        self.el = EdgeList()

    def get_vertices(self, index: int) -> List[Vertex]:
        vertices = []

        for point in self.get_single_data(index).points:
            vertices.append(self.vl.add(point))

        return vertices

    def test_find_existing(self):
        """Find an existing edge"""
        vertices = self.get_vertices(0)
        edge = self.el.add(vertices[0], vertices[1], Arc([0.5, 0.5, 0]))

        self.assertEqual(self.el.find(vertices[0], vertices[1]), edge)

    def test_find_existing_invertex(self):
        """Find an existing edge with inverted vertices"""
        vertices = self.get_vertices(0)
        edge = self.el.add(vertices[0], vertices[1], Arc([0.5, 0.5, 0]))

        self.assertEqual(self.el.find(vertices[1], vertices[0]), edge)

    def test_find_nonexisting(self):
        """Raise an EdgeNotFoundError for a non-existing edge"""
        vertices = self.get_vertices(0)
        edge = self.el.add(vertices[0], vertices[1], Arc([0.5, 0.5, 0]))

        with self.assertRaises(EdgeNotFoundError):
            self.assertEqual(self.el.find(vertices[1], vertices[2]), edge)

    def test_add_new(self):
        """Add an edge when no such thing exists"""
        vertices = self.get_vertices(0)
        self.el.add(vertices[0], vertices[1], Arc([0.5, 0.5, 0]))

        self.assertEqual(len(self.el.edges), 1)

    def test_add_existing(self):
        """Add the same edges twice"""
        vertices = self.get_vertices(0)
        self.el.add(vertices[0], vertices[1], Arc([0.5, 0.5, 0]))
        self.el.add(vertices[0], vertices[1], Arc([0.5, 0.5, 0]))

        self.assertEqual(len(self.el.edges), 1)

    def test_add_invalid(self):
        """Add a circular edge that's actually a line"""
        vertices = self.get_vertices(0)
        self.el.add(vertices[0], vertices[1], Arc([0.5, 0, 0]))

        self.assertEqual(len(self.el.edges), 0)

    def test_vertex_order(self):
        """Assure vertices maintain consistent order"""
        vertices = self.get_vertices(0)

        # add some custom edges
        self.el.add(vertices[2], vertices[3], Spline([[0.7, 1.3, 0], [0.3, 1.3, 0]]))
        self.el.add(vertices[6], vertices[7], PolyLine([[0.7, 1.1, 1], [0.3, 1.1, 1]]))

        self.assertEqual(len(self.el.edges), 2)

        self.assertEqual(self.el.edges[0].vertex_1.index, 2)
        self.assertEqual(self.el.edges[0].vertex_2.index, 3)

        self.assertEqual(self.el.edges[1].vertex_1.index, 6)
        self.assertEqual(self.el.edges[1].vertex_2.index, 7)

    def test_add_from_operations(self):
        """Add edges from an operation"""
        face = Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [None, Project("terrain"), None, None])

        revolve = Revolve(face, 1, [0, 0, 1], [-1, 0, 0])

        for point in revolve.point_array:
            self.vl.add(point)

        self.el.add_from_operation(self.vl.vertices, revolve)

        self.assertEqual(len(self.el.edges), 6)

        # 4 arcs from a revolve and 2 projections from faces,
        # 6 'line' edges
        no_arc = 0
        no_project = 0
    
        for edge in self.el.edges:
            if edge.kind == "angle":
                no_arc += 1
            elif edge.kind == "project":
                no_project += 1
        
        self.assertEqual(no_arc, 4)
        self.assertEqual(no_project, 2)

    def test_description(self):
        """Output for blockMesh"""
        vertices = self.get_vertices(0)
        self.el.add(vertices[0], vertices[1], Arc([0.5, 0.5, 0]))
        self.el.add(vertices[0], vertices[1], Arc([0.5, 0.5, 0]))

        expected = "edges\n(\n"

        for edge in self.el.edges:
            expected += edge.description + "\n"

        expected += ");\n\n"

        self.assertEqual(self.el.description, expected)