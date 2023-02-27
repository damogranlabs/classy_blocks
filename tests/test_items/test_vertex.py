import unittest

import numpy as np

from classy_blocks.items.vertex import Vertex

from classy_blocks.util import constants

class VertexTests(unittest.TestCase):
    def setUp(self):
        self.coords = [0, 0, 0]
    
    @property
    def vertex(self):
        return Vertex(self.coords)

    def test_assert_3d(self):
        """Raise an exception if the point is not in 3D space"""
        with self.assertRaises(AssertionError):
            self.coords = [0, 0]
            _ = self.vertex

    def test_translate_int(self):
        """Vertex translation with integer delta"""
        delta = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            self.vertex.translate(delta).pos, delta
        )
    
    def test_translate_float(self):
        """Vertex translation with float delta"""
        self.coords = [0.0, 0.0, 0.0]
        delta = [1.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(
            self.vertex.translate(delta).pos, delta
        )
    
    def test_rotate(self):
        """Rotate a point"""
        self.coords = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            self.vertex.rotate(np.pi/2, [0, 0, 1], [0, 0, 0]).pos,
            [0, 1, 0]
        )

    def test_scale(self):
        """Scale a point"""
        self.coords = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            self.vertex.scale(2, [0, 0, 0]).pos,
            [2, 0, 0]
        )
    
    def test_equal(self):
        """The __eq__ method returns True"""
        point_1 = Vertex([0, 0, 0])
        point_2 = Vertex([0, 0, 0 + constants.tol/10])
    
        self.assertTrue(point_1 == point_2)
    
    def test_inequal(self):
        """The __eq__ method returns False"""
        point_1 = Vertex([0, 0, 0])
        point_2 = Vertex([0, 0, 0 + 2*constants.tol])

        self.assertFalse(point_1 == point_2)