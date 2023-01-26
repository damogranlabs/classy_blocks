import unittest

import numpy as np

from classy_blocks.define.point import Point

class PointTests(unittest.TestCase):
    def setUp(self):
        self.coords = [0, 0, 0]
    
    @property
    def point(self):
        return Point(self.coords)
    
    def test_init_float(self):
        """Accept a list of floats and convert it to numpy array"""
        self.assertTrue(type(self.point.point) == type(np.array([0, 0, 0])))
    
    def test_assert_3d(self):
        """Raise an exception if the point is not in 3D space"""
        with self.assertRaises(AssertionError):
            self.coords = [0, 0]
            _ = self.point

    def test_translate_int(self):
        """Point translation with integer delta"""
        delta = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            self.point.translate(delta).point, delta
        )
    
    def test_translate_float(self):
        """Point translation with float delta"""
        self.coords = [0.0, 0.0, 0.0]
        delta = [1.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(
            self.point.translate(delta).point, delta
        )
    
    def test_rotate(self):
        """Rotate a point"""
        self.coords = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            self.point.rotate(np.pi/2, [0, 0, 1], [0, 0, 0]).point,
            [0, 1, 0]
        )

    def test_scale(self):
        """Scale a point"""
        self.coords = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            self.point.scale(2, [0, 0, 0]).point,
            [2, 0, 0]
        )