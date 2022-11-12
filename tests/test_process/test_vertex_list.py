import unittest

from classy_blocks.define.primitives import Vertex, Edge
from classy_blocks.util import functions as f
from classy_blocks.util import constants

from classy_blocks.process.lists.vertices import VertexList
from classy_blocks.process.lists.edges import EdgeList

from tests.fixtures import FixturedTestCase


class VertexListTests(FixturedTestCase):
    def test_collect_vertices_single(self):
        """Collect vertices from a single block"""
        vertices = VertexList()
        vertices.collect([self.block_0], [])

        self.assertEqual(len(vertices), 8)

    def test_collect_vertices_multiple(self):
        """Collect vertices from two touching blocks"""
        vertices = VertexList()
        vertices.collect([self.block_0, self.block_1], [])

        self.assertEqual(len(vertices), 12)
        
    def test_collect_vertices_indexes(self):
        """Check that the correct vertices are assigned to block
        on collect()"""
        vertices = VertexList()
        vertices.collect([self.block_0, self.block_1], [])

        # the second block should reuse some vertices
        first_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        second_indexes = [1, 8, 9, 2, 5, 10, 11, 6]

        self.assertListEqual([v.mesh_index for v in self.block_0.vertices], first_indexes)
        self.assertListEqual([v.mesh_index for v in self.block_1.vertices], second_indexes)

    def test_find(self):
        vertices = VertexList()
        vertices.collect([self.block_0, self.block_1], [])

        displacement = constants.tol/10

        for i, vertex in enumerate(vertices):
            # we're searching for this point
            point = vertex.point

            # but slightly displaced (well within tolerances)
            point = point + f.vector(displacement, displacement, displacement)

            self.assertEqual(vertices.find(point).mesh_index, i)

    def test_find_fail(self):
        vertices = VertexList()
        vertices.collect([self.block_0], [])

        self.assertIsNone(vertices.find(f.vector(999, 999, 999)))

    def test_output(self):
        vertices = VertexList()
        vertices.collect([self.block_0], [])
        self.maxDiff = None

        expected_output = "vertices\n(\n" + \
            "\t(0.00000000 0.00000000 0.00000000) // 0\n" + \
            "\t(1.00000000 0.00000000 0.00000000) // 1\n" + \
            "\t(1.00000000 1.00000000 0.00000000) // 2\n" + \
            "\t(0.00000000 1.00000000 0.00000000) // 3\n" + \
            "\t(0.00000000 0.00000000 1.00000000) // 4\n" + \
            "\t(1.00000000 0.00000000 1.00000000) // 5\n" + \
            "\t(1.00000000 1.00000000 1.00000000) // 6\n" + \
            "\t(0.00000000 1.00000000 1.00000000) // 7\n" + \
            ");\n\n"

        self.assertEqual(vertices.output(), expected_output)

class EdgeListTests(unittest.TestCase):
    def get_list(self, points):
        vertex_1 = Vertex([0, 0, 0])
        vertex_1.mesh_index = 0

        vertex_2 = Vertex([1, 0, 0])
        vertex_2.mesh_index = 1

        edge = Edge(0, 1, points)
        edge.vertex_1 = vertex_1
        edge.vertex_2 = vertex_2

        edge_list = EdgeList()
        edge_list.edges = [edge]

        return edge_list

    def test_straight_edge_format(self):
        edge_list = self.get_list(None)

        # do not include straight edges
        expected_output = "edges\n(\n);\n\n"

        self.assertEqual(expected_output, edge_list.output())

    def test_project_edge_point_list(self):
        edge_list = self.get_list("projected_face")

        self.assertEqual(edge_list.edges[0].type, 'project')
        self.assertTrue("0 1 (projected_face)" in edge_list.output())

    def test_arc_edge_format(self):
        """an Edge with a single given point is an arc edge and should be formatted as such"""
        point = [0, 0.25, 0]
        edge_list = self.get_list(point)

        e = edge_list.edges[0]

        self.assertEqual(e.type, "arc")
        self.assertListEqual(list(e.points), point)
        self.assertTrue("arc 0 1 (0.00000000 0.25000000 0.00000000)" in edge_list.output())

    def test_spline_edge_format(self):
        """if an edge is given with a list of points, it is a spline edge and should be
        formatted as such"""
        points = [
            [0.3, 0.1, 0],
            [0.5, 0.2, 0],
            [0.7, 0.1, 0],
        ]

        edge_list = self.get_list(points)
        e = edge_list.edges[0]

        self.assertEqual(e.type, "spline")
        self.assertTrue("spline 0 1 (" +\
            "(0.30000000 0.10000000 0.00000000) " + \
            "(0.50000000 0.20000000 0.00000000) " + \
            "(0.70000000 0.10000000 0.00000000))"
            in edge_list.output()
        )

    def test_wrong_edge_format(self):
        edge_list = self.get_list(None)
        edge_list.edges[0].type = "wrong"

        with self.assertRaises(Exception):
            edge_list.output()
