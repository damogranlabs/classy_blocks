import unittest

import numpy as np

from classy_blocks.cbtyping import NPPointType
from classy_blocks.construct.curves.analytic import AnalyticCurve
from classy_blocks.optimize.clamps.curve import CurveClamp, LineClamp, RadialClamp
from classy_blocks.optimize.clamps.free import FreeClamp
from classy_blocks.optimize.clamps.surface import ParametricSurfaceClamp, PlaneClamp
from classy_blocks.util import functions as f


class ClampTestsBase(unittest.TestCase):
    def setUp(self):
        self.position = np.array([0, 0, 0])


class FreeClampTests(ClampTestsBase):
    def test_free_init(self):
        """Initialization of FreeClamp"""
        clamp = FreeClamp(self.position)

        np.testing.assert_array_equal(clamp.params, self.position)

    def test_free_update(self):
        """Update params of a free clamp"""
        clamp = FreeClamp(self.position)
        clamp.update_params([1, 0, 0])

        np.testing.assert_array_equal(clamp.position, [1, 0, 0])


class CurveClampTests(ClampTestsBase):
    def setUp(self):
        super().setUp()

        def function(t: float) -> NPPointType:
            return f.vector(np.sin(t), np.cos(t), t)

        self.curve = AnalyticCurve(function, (0, 2 * np.pi))

    def test_line_init(self):
        """Initialization of LineClamp"""
        clamp = LineClamp(self.position, [0, 0, 0], [1, 1, 1])

        self.assertAlmostEqual(clamp.params[0], 0)

    def test_line_init_far(self):
        """Initialization that will yield t < 0"""
        clamp = LineClamp(self.position, [1, 1, 1], [2, 2, 2], (-100, 100))

        self.assertAlmostEqual(clamp.params[0], -(3**0.5))

    def test_line_value(self):
        clamp = LineClamp(self.position, [0, 0, 0], [1, 1, 1])

        clamp.update_params([3**0.5 / 2])

        np.testing.assert_array_almost_equal(clamp.position, [0.5, 0.5, 0.5])

    def test_line_bounds_lower(self):
        position = [-1, -1, -1]
        clamp = LineClamp(position, [0, 0, 0], [1, 1, 1], (0, 1))

        self.assertAlmostEqual(clamp.params[0], 0)

    def test_line_bounds_upper(self):
        position = [2, 2, 2]
        clamp = LineClamp(position, [0, 0, 0], [1, 1, 1], (0, 1))

        self.assertAlmostEqual(clamp.params[0], 1)

    def test_analytic_init(self):
        clamp = CurveClamp(self.position, self.curve)

        self.assertAlmostEqual(clamp.params[0], 0, places=3)

    def test_analytic_init_noncoincident(self):
        position = [0, 0, 1]
        clamp = CurveClamp(position, self.curve)

        self.assertAlmostEqual(clamp.params[0], 1, places=3)

    def test_analytic_bounds_lower(self):
        position = [-1, -1, -1]
        clamp = CurveClamp(position, self.curve)

        self.assertAlmostEqual(clamp.params[0], 0, places=3)

    def test_analytic_bounds_upper(self):
        position = [0, 0, 2]
        self.curve.bounds = (0, 1)
        clamp = CurveClamp(position, self.curve)

        self.assertAlmostEqual(clamp.params[0], 1, places=3)

    def test_radial_init(self):
        position = [1, 0, 0]
        clamp = RadialClamp(position, [0, 0, -1], [0, 0, 1])

        np.testing.assert_array_almost_equal(clamp.position, [1, 0, 0])

    def test_radial_rotate(self):
        position = [1, 0, 0]
        clamp = RadialClamp(position, [0, 0, -1], [0, 0, 1])

        clamp.update_params([np.pi / 2])

        np.testing.assert_array_almost_equal(clamp.position, [0, 1, 0])


class SurfaceClampTests(ClampTestsBase):
    def setUp(self):
        super().setUp()

        def function(params) -> NPPointType:
            """A simple extruded sinusoidal surface"""
            u = params[0]
            v = params[1]

            return f.vector(u, v, np.sin(u))

        self.function = function

    def test_plane_clamp(self):
        clamp = PlaneClamp(self.position, [0, 0, 0], [1, 1, 1])

        np.testing.assert_array_almost_equal(clamp.params, [0, 0])

    def test_plane_move_u(self):
        clamp = PlaneClamp(self.position, [0, 0, 0], [1, 0, 0])

        clamp.update_params([1, 0])

        self.assertAlmostEqual(clamp.position[0], 0)
        self.assertAlmostEqual(f.norm(clamp.position), 1)

    def test_plane_move_v(self):
        clamp = PlaneClamp(self.position, [0, 0, 0], [1, 0, 0])

        clamp.update_params([0, 1])

        self.assertAlmostEqual(clamp.position[0], 0)
        self.assertAlmostEqual(f.norm(clamp.position), 1)

    def test_plane_move_uv(self):
        clamp = PlaneClamp(self.position, [0, 0, 0], [1, 0, 0])

        clamp.update_params([1, 1])

        self.assertAlmostEqual(clamp.position[0], 0)
        self.assertAlmostEqual(f.norm(clamp.position), np.sqrt(2))

    def test_plane_basis_orthonormal(self):
        for normal in ([1, 0, 0], [0, 0, -1], [1, 1, 1], [-1, -1, -1], [1, 1e-9, 0], [0.3, -2, 0.7]):
            with self.subTest(normal=normal):
                clamp = PlaneClamp(self.position, [0, 0, 0], normal)
                basis = np.array([clamp.u_dir, clamp.v_dir, clamp.normal])

                np.testing.assert_array_almost_equal(basis @ basis.T, np.eye(3))

    def test_plane_oblique_stays_in_plane(self):
        point = np.array([1, 2, 3])
        normal = f.unit_vector([-1, -1, -1])
        clamp = PlaneClamp(self.position, point, normal)

        clamp.update_params([1.5, -2.5])

        self.assertAlmostEqual(np.dot(clamp.position - point, normal), 0)

    def test_plane_deterministic(self):
        clamp_1 = PlaneClamp(self.position, [0, 0, 0], [1, 1, 1])
        clamp_2 = PlaneClamp(self.position, [0, 0, 0], [1, 1, 1])

        np.testing.assert_array_equal(clamp_1.u_dir, clamp_2.u_dir)
        np.testing.assert_array_equal(clamp_1.v_dir, clamp_2.v_dir)

    def test_parametric_init(self):
        clamp = ParametricSurfaceClamp(self.position, self.function)

        np.testing.assert_array_almost_equal(clamp.params, [0, 0], decimal=3)

    def test_parametric_initial_unbounded(self):
        clamp = ParametricSurfaceClamp(self.position, self.function)

        np.testing.assert_array_almost_equal(clamp.initial_guess, [0, 0])

    def test_parametric_initial_bounded(self):
        clamp = ParametricSurfaceClamp(self.position, self.function, [[0, 1], [0, 1]])

        np.testing.assert_array_almost_equal(clamp.initial_guess, [0, 0])

    def test_parametric_move(self):
        clamp = ParametricSurfaceClamp(self.position, self.function)

        clamp.update_params([np.pi / 2, 1])

        np.testing.assert_array_almost_equal(clamp.position, [np.pi / 2, 1, 1])

    def test_parametric_bounds_upper(self):
        position = [4, 4, 0]
        clamp = ParametricSurfaceClamp(position, self.function, [[0.0, np.pi], [0.0, np.pi]])

        np.testing.assert_array_almost_equal(clamp.params, [np.pi, np.pi])
