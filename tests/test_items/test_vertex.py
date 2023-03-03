import numpy as np

from classy_blocks.base.exceptions import VertexNotFoundError
from classy_blocks.items.vertex import Vertex
from classy_blocks.util import constants
from classy_blocks.util import functions as f

from tests.fixtures.data import DataTestCase

class VertexTests(DataTestCase):
    def setUp(self):    
    def test_assert_3d(self):
        """Raise an exception if the point is not in 3D space"""
        with self.assertRaises(AssertionError):
            _ = Vertex([0, 0], 0)

    def test_translate_int(self):
        """Vertex translation with integer delta"""
        delta = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            Vertex([0, 0, 0], 0).translate(delta).pos, delta
        )
    
    def test_translate_float(self):
        """Vertex translation with float delta"""
        coords = [0.0, 0.0, 0.0]
        delta = [1.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(
            Vertex(coords, 0).translate(delta).pos, delta
        )
    
    def test_rotate(self):
        """Rotate a point"""
        coords = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            Vertex(coords, 0).rotate(np.pi/2, [0, 0, 1], [0, 0, 0]).pos,
            [0, 1, 0]
        )

    def test_scale(self):
        """Scale a point"""
        coords = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            Vertex(coords, 0).scale(2, [0, 0, 0]).pos,
            [2, 0, 0]
        )

    def test_inequal(self):
        """The __eq__ method returns False"""
        point_1 = Vertex([0, 0, 0], 0)
        point_2 = Vertex([0, 0, 0 + 2*constants.TOL], 1)

        self.assertFalse(point_1 == point_2)

