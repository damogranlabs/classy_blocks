import unittest

import numpy as np

from classy_blocks.items.vertex import Vertex

from classy_blocks.items.edges.arcs.arc import ArcEdge
from classy_blocks.items.edges.arcs.origin import OriginEdge, arc_from_origin
from classy_blocks.items.edges.arcs.angle import AngleEdge, arc_from_theta
from classy_blocks.items.edges.spline import SplineEdge, PolyLineEdge
from classy_blocks.items.edges.project import ProjectEdge
from classy_blocks.items.edges.factory import factory

from classy_blocks.util import functions as f

class EdgeTests(unittest.TestCase):
    def setUp(self):
        Vertex.registry = []
        factory.registry = []

    def test_assert_vertices(self):
        """Make sure vertices aren't passed as numpy points or whatever"""
        with self.assertRaises(AssertionError):
            _ = ArcEdge([0, 0, 0], [1, 0, 0], [0.5, 0.2, 0])

    def test_arc_edge_translate(self):
        arc_edge = ArcEdge(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            [0.5, 0, 0]
        )
        arc_edge.translate([1, 1, 1])
        np.testing.assert_array_equal(arc_edge.arc_point, [1.5, 1, 1])

    def test_angle_edge_rotate_1(self):
        angle_edge = AngleEdge(
            [0, 0, 0], [1, 0, 0],
            np.pi/2, [0, 0, 1]
        )

        angle_edge.rotate(np.pi/2, [0, 0, 1], [0, 0, 0])

        np.testing.assert_array_equal(
            angle_edge.axis, [0, 0, 1])

        self.assertEqual(angle_edge.angle, np.pi/2)

    def test_angle_edge_rotate_2(self):
        angle_edge = AngleEdge(
            [0, 0, 0], [1, 0, 0],
            np.pi/2, [0, 0, 1]
        )

        angle_edge.rotate(np.pi/2, [1, 0, 0], [0, 0, 0])

        np.testing.assert_array_equal(
            angle_edge.axis, [0, -1, 0])

        self.assertEqual(angle_edge.angle, np.pi/2)

    def test_spline_edge_translate(self):
        spline_edge = SplineEdge(
            [0, 0, 0], [1, 0, 0],
            [
                [0.25, 0.1, 0],
                [0.5, 0.5, 0],
                [0.75, 0.1, 0],
            ]
        )

        spline_edge.translate([1, 1, 1])
    
        np.testing.assert_array_equal(
            spline_edge.points, [
                [1.25, 1.1, 1],
                [1.5, 1.5, 1],
                [1.75, 1.1, 1],
            ]
        )

class EdgeFactoryTests(unittest.TestCase):
    """Factory tests; examples from BlockDef.add_edge docstring"""
    def setUp(self):
        Vertex.registry = []
        factory.registry = []

    def test_arc(self):
        arc_point = [0.5, 0.2, 0]

        edg = factory.create(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            'arc', arc_point)

        self.assertIsInstance(edg, ArcEdge)
        self.assertListEqual(arc_point, edg.arc_point)

    def test_default_origin(self):
        origin = [0.5, -0.5, 0]
        flatness = 2
        edg = factory.create(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            'origin', origin, flatness)

        self.assertIsInstance(edg, OriginEdge)
        self.assertListEqual(origin, edg.origin)
        self.assertEqual(flatness, edg.flatness)

    def test_flat_origin(self):
        origin = [0.5, -0.5, 0]
        edg = factory.create(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            'origin', origin)

        self.assertIsInstance(edg, OriginEdge)
        self.assertListEqual(origin, edg.origin)
        self.assertEqual(1, edg.flatness)

    def test_angle(self):
        angle = np.pi/6
        axis = [0, 0, 1]
        edg = factory.create(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            'angle', angle, axis)

        self.assertIsInstance(edg, AngleEdge)
        self.assertEqual(angle, edg.angle)
        self.assertListEqual(axis, edg.axis)

    def test_spline(self):
        points = [[0.3, 0.25, 0], [0.6, 0.1, 0], [0.3, 0.25, 0]]
        edg = factory.create(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            'spline', points)

        self.assertIsInstance(edg, SplineEdge)
        self.assertListEqual(points, edg.points)

    def test_polyline_edge(self):
        points = [[0.3, 0.25, 0], [0.6, 0.1, 0], [0.3, 0.25, 0]]

        edg = factory.create(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            'polyLine', points)

        self.assertIsInstance(edg, PolyLineEdge)
        self.assertIsInstance(edg, SplineEdge)
        self.assertListEqual(points, edg.points)

    def test_projected_edge(self):
        geometry = 'terrain'
        edg = factory.create(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            'project', geometry)

        self.assertIsInstance(edg, ProjectEdge)
        self.assertEqual(edg.geometry, geometry)

    def test_create_duplicate(self):
        """Create two edges between the same pair of vertices"""
        edge_1 = factory.create(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            'arc', [0.5, 0.2, 0])
    
        edge_2 = factory.create(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            'project', 'terrain')
        
        self.assertEqual(id(edge_1), id(edge_2))
        self.assertEqual(edge_1.kind, 'arc')
    
    def test_create_duplicate_inverted(self):
        """Create two edges between the same pair of vertices, inverted"""
        edge_1 = factory.create(
            Vertex([0, 0, 0]),
            Vertex([1, 0, 0]),
            'arc', [0.5, 0.2, 0])
    
        edge_2 = factory.create(
            Vertex([1, 0, 0]),
            Vertex([0, 0, 0]),
            'project', 'terrain')
        
        self.assertEqual(id(edge_1), id(edge_2))
        self.assertEqual(edge_1.kind, 'arc')

# class TestPrimitives(unittest.TestCase):
#     def setUp(self):
#         v1 = Vertex([0, 0, 0])
#         v1.mesh_index = 0

#         v2 = Vertex([1, 0, 0])
#         v2.mesh_index = 1

#         self.v1 = v1
#         self.v2 = v2

#     def make_edge(self, points, check_valid=True, kind=None):
#         e = Edge(0, 1, points, kind)
#         e.vertex_1 = self.v1
#         e.vertex_2 = self.v2

#         if check_valid:
#             self.assertTrue(e.is_valid)

#         return e

#     def test_edge_validity(self):
#         """arc edges must always be checked for validity, spline edges never"""
#         # spline edge: all are valid, even collinear
#         p = [
#             [0.3, 0, 0],
#             [0.7, 0, 0],
#         ]
#         self.make_edge(p)

#         # points collinear with line v1-v2 do not make a valid edge
#         p = [0.5, 0, 0]
#         e = self.make_edge(p, check_valid=False)

#         self.assertFalse(e.is_valid)

#         # other points do make a valid edge
#         p = [0.5, 0.2, 0]
#         e = self.make_edge(p)

#     def test_straight_edge_fail(self):
#         """Straight edges are not needed"""
#         with self.assertRaises(AssertionError):
#             e = self.make_edge(None)

#     def test_projected_edge_length(self):
#         e = self.make_edge("projected_geometry")
#         self.assertAlmostEqual(e.get_length(), 1)

#     def test_arc_edge_length(self):
#         p = [0.5, 0.5, 0]
#         e = self.make_edge(p)

#         self.assertAlmostEqual(e.get_length(), np.pi / 2)

#     def test_spline_edge_length(self):
#         p = [[0, 1, 0], [1, 1, 0]]

#         e = self.make_edge(p)

#         self.assertAlmostEqual(e.get_length(), 3)

#     def test_arc_edge_kind(self):
#         e = self.make_edge([0.5, 0.2, 0], check_valid=False)
#         self.assertEqual(e.kind, "arc")

#     def test_spline_edge_kind(self):
#         e = self.make_edge(
#             [
#                 [0.25, 0.1, 0],
#                 [0.5, 0.2, 0],
#                 [0.75, 0.1, 0],
#             ],
#             check_valid=False,
#         )
#         self.assertEqual(e.kind, "spline")

#     def test_project_edge_kind(self):
#         e = self.make_edge("projected_face", check_valid=False)
#         self.assertEqual(e.kind, "project")

#     def test_specified_edge_kind(self):
#         e = self.make_edge([
#             [0.25, 0.1, 0],
#             [0.5, 0.2, 0],
#             [0.75, 0.1, 0],
#         ], kind='polyLine')
        
#         self.assertEqual(e.kind, 'polyLine')

#     def test_rotate_project_edge(self):
#         e = self.make_edge("test_geometry")
#         r = e.rotate(1, [1, 0, 0])
#         self.assertEqual(r.points, "test_geometry")

#     def test_rotate_arc_edge(self):
#         point = [0.5, 0.1, 0]
#         angle = 1

#         e = self.make_edge(point)
#         r = e.rotate(angle, axis=[1, 0, 0])
#         np.testing.assert_array_almost_equal(r.points, f.rotate(point, angle, axis="x"))

#     def test_rotate_spline_edge(self):
#         points = [
#             [0.25, 0.1, 0],
#             [0.5, 0.2, 0],
#             [0.75, 0.1, 0],
#         ]
#         angle = 1

#         e = self.make_edge(points, check_valid=False)
#         r = e.rotate(angle, axis=[1, 0, 0])

#         np.testing.assert_array_almost_equal(r.points, [f.rotate(p, angle, axis="x") for p in points])

class AlternativeArcTests(unittest.TestCase):
    unit_sq_corner = f.vector(2**0.5/2, 2**0.5/2, 0)

    def test_arc_mid(self):
        axis = f.vector(0, 0, 1)
        center = f.vector(0, 0, 0)
        radius = 1
        edge_point_1 = f.vector(1, 0, 0)
        edge_point_2 = f.vector(0, 1, 0)

        np.testing.assert_array_almost_equal(
            f.arc_mid(axis, center, radius, edge_point_1, edge_point_2),
            self.unit_sq_corner
        )

    def test_arc_from_theta(self):
        edge_point_1 = f.vector(0, 1, 0)
        edge_point_2 = f.vector(1, 0, 0)
        angle = np.pi/2
        axis = f.vector(0, 0, -1)

        np.testing.assert_array_almost_equal(
            arc_from_theta(edge_point_1, edge_point_2, angle, axis),
            self.unit_sq_corner
        )

    def test_arc_from_origin(self):
        edge_point_1 = f.vector(0, 1, 0)
        edge_point_2 = f.vector(1, 0, 0)
        center = f.vector(0, 0, 0)

        np.testing.assert_array_almost_equal(
            arc_from_origin(edge_point_1, edge_point_2, center),
            self.unit_sq_corner
        )

    def test_arc_from_origin_warn(self):
        edge_point_1 = f.vector(0, 1, 0)
        edge_point_2 = f.vector(1.1, 0, 0)
        center = f.vector(0, 0, 0)

        with self.assertWarns(Warning):
            adjusted_point = arc_from_origin(edge_point_1, edge_point_2, center)
        
        expected_point = f.vector(0.75743894, 0.72818283, 0)
        
        np.testing.assert_array_almost_equal(
            adjusted_point,
            expected_point
        )
