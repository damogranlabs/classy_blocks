import unittest

import numpy as np
from classy_blocks.define.primitives import Vertex, Edge, WrongEdgeTypeException
from classy_blocks.util import functions as f
from classy_blocks.util import constants


class TestPrimitives(unittest.TestCase):
    def setUp(self):
        v1 = Vertex([0, 0, 0])
        v1.mesh_index = 0

        v2 = Vertex([1, 0, 0])
        v2.mesh_index = 1

        self.v1 = v1
        self.v2 = v2

    def make_edge(self, points, check_valid=True):
        e = Edge(0, 1, points)
        e.vertex_1 = self.v1
        e.vertex_2 = self.v2

        if check_valid:
            self.assertTrue(e.is_valid)

        return e

    

    def test_edge_validity(self):
        """arc edges must always be checked for validity, spline edges never"""
        # spline edge: all are valid, even collinear
        p = [
            [0.3, 0, 0],
            [0.7, 0, 0],
        ]
        self.make_edge(p)

        # points collinear with line v1-v2 do not make a valid edge
        p = [0.5, 0, 0]
        e = self.make_edge(p, check_valid=False)

        self.assertFalse(e.is_valid)

        # other points do make a valid edge
        p = [0.5, 0.2, 0]
        e = self.make_edge(p)

    def test_straight_edge_length(self):
        e = self.make_edge(None)
        self.assertAlmostEqual(e.get_length(), 1)

    def test_projected_edge_length(self):
        e = self.make_edge("projected_geometry")
        self.assertAlmostEqual(e.get_length(), 1)

    def test_arc_edge_length(self):
        p = [0.5, 0.5, 0]
        e = self.make_edge(p)

        self.assertAlmostEqual(e.get_length(), np.pi / 2)

    def test_spline_edge_length(self):
        p = [[0, 1, 0], [1, 1, 0]]

        e = self.make_edge(p)

        self.assertAlmostEqual(e.get_length(), 3)

    def test_straight_edge_type(self):
        e = self.make_edge(None, check_valid=False)
        self.assertEqual(e.type, "line")

    def test_arc_edge_type(self):
        e = self.make_edge([0.5, 0.2, 0], check_valid=False)
        self.assertEqual(e.type, "arc")

    def test_spline_edge_type(self):
        e = self.make_edge(
            [
                [0.25, 0.1, 0],
                [0.5, 0.2, 0],
                [0.75, 0.1, 0],
            ],
            check_valid=False,
        )
        self.assertEqual(e.type, "spline")

    def test_project_edge_type(self):
        e = self.make_edge("projected_face", check_valid=False)
        self.assertEqual(e.type, "project")

    def test_wrong_edge_type(self):
        with self.assertRaises(Exception):
            e = self.make_edge(None)
            e.type = "wrong"
            e.point_list

        with self.assertRaises(Exception):
            e = self.make_Edge(None)
            e.type = "wrong"
            e.get_length()

    def test_rotate_straight_edge(self):
        e = self.make_edge(None)
        r = e.rotate(1)
        self.assertEqual(r.points, None)

    def test_rotate_project_edge(self):
        e = self.make_edge("test_geometry")
        r = e.rotate(1)
        self.assertEqual(r.points, "test_geometry")

    def test_rotate_arc_edge(self):
        point = [0.5, 0.1, 0]
        angle = 1

        e = self.make_edge(point)
        r = e.rotate(angle, axis=[1, 0, 0])
        np.testing.assert_array_almost_equal(r.points, f.rotate(point, angle, axis="x"))

    def test_rotate_spline_edge(self):
        points = [
            [0.25, 0.1, 0],
            [0.5, 0.2, 0],
            [0.75, 0.1, 0],
        ]
        angle = 1

        e = self.make_edge(points, check_valid=False)
        r = e.rotate(angle, axis=[1, 0, 0])

        np.testing.assert_array_almost_equal(r.points, [f.rotate(p, angle, axis="x") for p in points])

    def test_rotate_wrong_edge(self):
        e = self.make_edge(None)
        e.type = "wrong"

        with self.assertRaises(Exception):
            e.rotate(1)


if __name__ == "__main__":
    unittest.main()
