import unittest

import numpy as np

from classy_blocks.classes.primitives import Vertex, Edge
from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.operations import Face

from classy_blocks.util import constants

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
            Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0.1]], check_coplanar=True)

if __name__ == '__main__':
    unittest.main()