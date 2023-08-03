import unittest

import numpy as np

from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.clamps.curve import LineClamp
from classy_blocks.modify.clamps.free import FreeClamp
from classy_blocks.modify.clamps.surface import PlaneClamp


class ClampTests(unittest.TestCase):
    def setUp(self):
        self.vertex = Vertex([0, 0, 0], 0)

    def test_free_init(self):
        """Initialization of FreeClamp"""
        clamp = FreeClamp(self.vertex)

        np.testing.assert_array_equal(clamp.params, self.vertex.position)

    def test_free_update(self):
        """Update params of a free clamp"""
        clamp = FreeClamp(self.vertex)
        clamp.update_params([1, 0, 0])

        np.testing.assert_array_equal(self.vertex.position, [1, 0, 0])

    def test_line_init(self):
        """Initialization of LineClamp"""
        clamp = LineClamp(self.vertex, [0, 0, 0], [1, 1, 1])

        self.assertAlmostEqual(clamp.params[0], 0)

    def test_line_init_warning_result(self):
        """Initialization of LineClamp with a non-coincident vertex;
        update vertex with closest point"""
        clamp = LineClamp(self.vertex, [1, 1, 1], [2, 1, 1])

        # don't be too strict about initial parameters,
        # optimization will move everything away anyhow
        np.testing.assert_array_almost_equal(clamp.point, [0, 1, 1], decimal=3)

    def test_line_init_far(self):
        """Initialization that will yield t < 0"""
        clamp = LineClamp(self.vertex, [1, 1, 1], [2, 2, 2])

        self.assertAlmostEqual(clamp.params[0], -1)

    def test_line_value(self):
        clamp = LineClamp(self.vertex, [0, 0, 0], [1, 1, 1])

        clamp.update_params([0.5])

        np.testing.assert_array_almost_equal(clamp.point, [0.5, 0.5, 0.5])

    def test_plane_clamp(self):
        clamp = PlaneClamp(self.vertex, [0, 0, 0], [1, 1, 1])

        np.testing.assert_array_almost_equal(clamp.params, [0, 0])

    def test_plane_move_u(self):
        clamp = PlaneClamp(self.vertex, [0, 0, 0], [1, 0, 0])

        clamp.update_params([1, 0])

        self.assertAlmostEqual(self.vertex.position[0], 0)

    def test_plane_move_v(self):
        clamp = PlaneClamp(self.vertex, [0, 0, 0], [1, 0, 0])

        clamp.update_params([0, 1])

        self.assertAlmostEqual(self.vertex.position[0], 0)

    def test_plane_move_uv(self):
        clamp = PlaneClamp(self.vertex, [0, 0, 0], [1, 0, 0])

        clamp.update_params([1, 1])

        self.assertAlmostEqual(self.vertex.position[0], 0)
