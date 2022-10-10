import unittest

import numpy as np
from classy_blocks.classes.primitives import Vertex, Edge, WrongEdgeTypeException
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

    def test_vertex_output(self):
        """check Vertex() output"""
        v = Vertex([1.2, 2.05, 2 / 3])

        expected_output = "(1.20000000 2.05000000 0.66666667)"

        # a vertex without mesh_index is just a point
        self.assertEqual(str(v), expected_output)

        # add a mesh index and expect it along expected_output
        v.mesh_index = 2
        self.assertEqual(str(v), expected_output + " // 2")

    def test_straight_edge_format(self):
        e = self.make_edge(None, check_valid=False)
        self.assertIsNone(e.point_list)
        self.assertEqual(str(e), "line 0 1 None")

    def test_project_edge_point_list(self):
        e = self.make_edge("projected_face", check_valid=False)
        self.assertEqual(e.point_list, "(projected_face)")
        self.assertEqual(str(e), "project 0 1 (projected_face)")

    def test_arc_edge_format(self):
        """an Edge with a single given point is an arc edge and should be formatted as such"""
        p = [0, 0.25, 0]

        e = self.make_edge(p)

        self.assertEqual(e.type, "arc")
        self.assertListEqual(list(e.points), list(np.array(p)))
        self.assertEqual(str(e), "arc 0 1 (0.00000000 0.25000000 0.00000000)")

    def test_spline_edge_format(self):
        """if an edge is given with a list of points, it is a spline edge and should be
        formatted as such"""
        p = [
            [0.3, 0.1, 0],
            [0.5, 0.2, 0],
            [0.7, 0.1, 0],
        ]

        e = self.make_edge(p)

        self.assertEqual(e.type, "spline")
        self.assertEqual(
            str(e),
            "spline 0 1 ("
            "(0.30000000 0.10000000 0.00000000) "
            "(0.50000000 0.20000000 0.00000000) "
            "(0.70000000 0.10000000 0.00000000)"
            ")",
        )

    def test_wrong_edge_format(self):
        e = self.make_edge(None)
        e.type = "wront"

        with self.assertRaises(Exception):
            str(e)

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
