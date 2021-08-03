import unittest

import numpy as np
from classy_blocks.classes.primitives import Vertex, Edge

class TestPrimitives(unittest.TestCase):
    def test_vertex_output(self):
        """ check Vertex() output """
        v = Vertex([1.2, 2.05, 2/3])

        expected_output = "(1.20000000 2.05000000 0.66666667)"

        # a vertex without mesh_index is just a point
        self.assertEqual(str(v), expected_output)

        # add a mesh index and expect it along expected_output
        v.mesh_index = 2
        self.assertEqual(str(v), expected_output +  " // 2")
    
    def test_arc_edge(self):
        """ an Edge with a single given point is an arc edge and should be formatted as such """
        v1 = Vertex([0, 0, 0])
        v1.mesh_index = 0

        v2 = Vertex([1, 0, 0])
        v2.mesh_index = 1
        p = [0.5, 0, 0]

        e = Edge(0, 1, p)
        e.vertex_1 = v1
        e.vertex_2 = v2

        self.assertEqual(e.type, 'arc')
        self.assertListEqual(list(e.points), list(np.array(p)))
        self.assertEqual(str(e), "arc 0 1 (0.50000000 0.00000000 0.00000000)")

    def test_spline_edge(self):
        """ if an edge is given with a list of points, it is a spline edge and should be
        formatted as such """
        v1 = Vertex([0, 0, 0])
        v1.mesh_index = 0

        v2 = Vertex([1, 0, 0])
        v2.mesh_index = 1

        p = [
            [0.3, 0.1, 0],
            [0.5, 0.2, 0],
            [0.7, 0.1, 0],
        ]

        e = Edge(0, 1, p)
        e.vertex_1 = v1
        e.vertex_2 = v2

        self.assertEqual(e.type, 'spline')
        self.assertEqual(str(e), 
            "spline 0 1 ("
                "(0.30000000 0.10000000 0.00000000) "
                "(0.50000000 0.20000000 0.00000000) "
                "(0.70000000 0.10000000 0.00000000)"
            ")"
        )

    def test_edge_valid(self):
        """ arc edges must be checked for validity, spline edges never """
        v1 = Vertex([0, 0, 0])
        v1.mesh_index = 0

        v2 = Vertex([1, 0, 0])
        v2.mesh_index = 1

        # spline edge: all are valid, even collinear
        p = [ [0.3, 0, 0],  [0.7, 0, 0], ]
        e = Edge(0, 1, p)
        
        self.assertTrue(e.is_valid)

        # points collinear with line v1-v2 do not make a valid edge
        p = [0.5, 0, 0]
        e = Edge(0, 1, p)
        e.vertex_1 = v1
        e.vertex_2 = v2

        self.assertFalse(e.is_valid)
        
        # other points do make a valid edge
        p = [0.5, 0.2, 0]
        e = Edge(0, 1, p)
        e.vertex_1 = v1
        e.vertex_2 = v2

        self.assertTrue(e.is_valid)

if __name__ == '__main__':
    unittest.main()