import unittest

import numpy as np
from classy_blocks.classes.primitives import Vertex, Edge

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
        """ check Vertex() output """
        v = Vertex([1.2, 2.05, 2/3])

        expected_output = "(1.20000000 2.05000000 0.66666667)"

        # a vertex without mesh_index is just a point
        self.assertEqual(str(v), expected_output)

        # add a mesh index and expect it along expected_output
        v.mesh_index = 2
        self.assertEqual(str(v), expected_output +  " // 2")
    
    def test_arc_edge_format(self):
        """ an Edge with a single given point is an arc edge and should be formatted as such """
        p = [0, 0.25, 0]

        e = self.make_edge(p)
        
        self.assertEqual(e.type, 'arc')
        self.assertListEqual(list(e.points), list(np.array(p)))
        self.assertEqual(str(e), "arc 0 1 (0.00000000 0.25000000 0.00000000)")

    def test_spline_edge_format(self):
        """ if an edge is given with a list of points, it is a spline edge and should be
        formatted as such """
        p = [
            [0.3, 0.1, 0],
            [0.5, 0.2, 0],
            [0.7, 0.1, 0],
        ]

        e = self.make_edge(p)
        
        self.assertEqual(e.type, 'spline')
        self.assertEqual(str(e), 
            "spline 0 1 ("
                "(0.30000000 0.10000000 0.00000000) "
                "(0.50000000 0.20000000 0.00000000) "
                "(0.70000000 0.10000000 0.00000000)"
            ")"
        )

    def test_edge_validity(self):
        """ arc edges must be checked for validity, spline edges never """
        # spline edge: all are valid, even collinear
        p = [ [0.3, 0, 0],  [0.7, 0, 0], ]
        self.make_edge(p)

        # points collinear with line v1-v2 do not make a valid edge
        p = [0.5, 0, 0]
        e = self.make_edge(p, check_valid=False)
        
        self.assertFalse(e.is_valid)
        
        # other points do make a valid edge
        p = [0.5, 0.2, 0]
        e = self.make_edge(p)

    def test_arc_edge_length(self):
        p = [0.5, 0.5, 0]
        e = self.make_edge(p)

        # although it's a circular edge,
        # approximate length calculation is the same for arc and spline edges;
        # just a sum of straight lines between points
        self.assertAlmostEqual(
            e.get_length(),
            2**0.5
        )
    
    def test_spline_edge_length(self):
        p = [
            [0, 1, 0],
            [1, 1, 0]
        ]

        e = self.make_edge(p)

        self.assertAlmostEqual(
            e.get_length(),
            3
        )


if __name__ == '__main__':
    unittest.main()