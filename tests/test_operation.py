import unittest

import numpy as np
from classes.primitives import Vertex, Edge
from classes.mesh import Mesh

from operations.operations import Face

from util import constants

from tests.fixtures import FixturedTestCase

class FaceTests(unittest.TestCase):
    def test_face_points(self):
        # provide less than 4 points
        with self.assertRaises(Exception):
            Face([
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0]
            ])
    
    def test_face_edges(self):
        with self.assertRaises(Exception):
            Face(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
                [None, None, None]
            )

    def test_coplanar_points(self):
        with self.assertRaises(Exception):
            Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.1]])

if __name__ == '__main__':
    unittest.main()