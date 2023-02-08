import unittest

from classy_blocks.process.items.vertex import Vertex
from classy_blocks.process.items.edge_ops import Edge
from classy_blocks.process.lists.edge_list import EdgeList

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

    def test_project_edge_point_list(self):
        edge_list = self.get_list("projected_face")

        self.assertEqual(edge_list.edges[0].kind, 'project')
        self.assertTrue("0 1 (projected_face)" in edge_list.output())

    def test_arc_edge_format(self):
        """an Edge with a single given point is an arc edge and should be formatted as such"""
        point = [0, 0.25, 0]
        edge_list = self.get_list(point)

        e = edge_list.edges[0]

        self.assertEqual(e.kind, "arc")
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

        self.assertEqual(e.kind, "spline")
        self.assertTrue("spline 0 1 (" +\
            "(0.30000000 0.10000000 0.00000000) " + \
            "(0.50000000 0.20000000 0.00000000) " + \
            "(0.70000000 0.10000000 0.00000000))"
            in edge_list.output()
        )
