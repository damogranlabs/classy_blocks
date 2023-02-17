import unittest

import numpy as np

from classy_blocks.data.point import Point

from classy_blocks.util import constants

class PointTests(unittest.TestCase):
    def setUp(self):
        self.coords = [0, 0, 0]
    
    @property
    def point(self):
        return Point(self.coords)

    def test_assert_3d(self):
        """Raise an exception if the point is not in 3D space"""
        with self.assertRaises(AssertionError):
            self.coords = [0, 0]
            _ = self.point

    def test_translate_int(self):
        """Point translation with integer delta"""
        delta = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            self.point.translate(delta).pos, delta
        )
    
    def test_translate_float(self):
        """Point translation with float delta"""
        self.coords = [0.0, 0.0, 0.0]
        delta = [1.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(
            self.point.translate(delta).pos, delta
        )
    
    def test_rotate(self):
        """Rotate a point"""
        self.coords = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            self.point.rotate(np.pi/2, [0, 0, 1], [0, 0, 0]).pos,
            [0, 1, 0]
        )

    def test_scale(self):
        """Scale a point"""
        self.coords = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            self.point.scale(2, [0, 0, 0]).pos,
            [2, 0, 0]
        )
    
    def test_equal(self):
        """The __eq__ method returns True"""
        point_1 = Point([0, 0, 0])
        point_2 = Point([0, 0, 0 + constants.tol/10])
    
        self.assertTrue(point_1 == point_2)
    
    def test_inequal(self):
        """The __eq__ method returns False"""
        point_1 = Point([0, 0, 0])
        point_2 = Point([0, 0, 0 + 2*constants.tol])

        self.assertFalse(point_1 == point_2)