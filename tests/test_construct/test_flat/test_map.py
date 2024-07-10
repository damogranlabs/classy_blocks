import unittest

import numpy as np

from classy_blocks.construct.flat.quad import Quad
from classy_blocks.util import functions as f


class QuadTests(unittest.TestCase):
    def setUp(self):
        self.positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])

    @property
    def quad(self) -> Quad:
        return Quad(self.positions, [0, 1, 2, 3])

    def test_quad_update(self):
        quad = self.quad
        new_position = [-1, -1, 0]

        self.positions[0] = new_position
        quad.update(self.positions)

        np.testing.assert_equal(quad.points, self.positions)

    def test_contains(self):
        self.assertTrue(self.quad.contains(f.vector(0, 0, 0)))

    def test_not_contains(self):
        self.assertFalse(self.quad.contains(f.vector(0, 0, 1)))

    def test_perimeter_square(self):
        self.assertEqual(self.quad.perimeter, 4)

    def test_perimeter_rectangle(self):
        delta = f.vector(1, 0, 0)
        self.positions[1] += delta
        self.positions[2] += delta

        self.quad.update(self.positions)

        self.assertEqual(self.quad.perimeter, 6)

    def test_center(self):
        np.testing.assert_equal(self.quad.center, [0.5, 0.5, 0])

    def test_e1(self):
        np.testing.assert_almost_equal(self.quad.e1, [1, 0, 0])

    def test_e2(self):
        np.testing.assert_almost_equal(self.quad.e2, [0, 1, 0])

    def test_normal(self):
        np.testing.assert_almost_equal(self.quad.normal, [0, 0, 1])
