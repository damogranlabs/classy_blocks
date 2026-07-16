import unittest

import numpy as np

from classy_blocks.construct.curves.interpolated import SplineInterpolatedCurve
from classy_blocks.construct.surfaces.surface import SurfaceBase


class SortPointsTests(unittest.TestCase):
    def test_empty_returns_empty(self):
        result = SurfaceBase.sort_points([], far_point=[0, 0, 0])
        self.assertEqual(len(result), 0)

    def test_single_loop_starts_nearest_far_point(self):
        # a straight run of points along +x, deliberately out of order
        loop = np.array([[2, 0, 0], [0, 0, 0], [3, 0, 0], [1, 0, 0]])

        result = SurfaceBase.sort_points([loop], far_point=[-10, 0, 0])

        np.testing.assert_allclose(result, [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])

    def test_far_point_flips_direction(self):
        loop = np.array([[2, 0, 0], [0, 0, 0], [3, 0, 0], [1, 0, 0]])

        result = SurfaceBase.sort_points([loop], far_point=[10, 0, 0])

        np.testing.assert_allclose(result, [[3, 0, 0], [2, 0, 0], [1, 0, 0], [0, 0, 0]])

    def test_joins_two_loops(self):
        loop_1 = np.array([[0, 0, 0], [1, 0, 0]])
        loop_2 = np.array([[3, 0, 0], [2, 0, 0]])

        result = SurfaceBase.sort_points([loop_1, loop_2], far_point=[-10, 0, 0])

        self.assertEqual(len(result), 4)
        np.testing.assert_allclose(result, [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])


class SortPointsMergeTests(unittest.TestCase):
    def setUp(self):
        # two closed square loops, each with a coincident closing vertex
        # (first == last), exactly as trimesh's section().discrete returns them
        self.loop_1 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
        self.loop_2 = np.array([[5, 0, 0], [6, 0, 0], [6, 1, 0], [5, 1, 0], [5, 0, 0]])

    def test_default_removes_coincident_closing_vertices(self):
        path = SurfaceBase.sort_points([self.loop_1, self.loop_2], far_point=[-10, 0, 0])

        segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
        self.assertTrue((segments > 0).all(), "joined path still has zero-length segments")

    def test_joined_path_builds_a_spline(self):
        path = SurfaceBase.sort_points([self.loop_1, self.loop_2], far_point=[-10, 0, 0])

        # must not raise ValueError: Expect x to not have duplicates
        SplineInterpolatedCurve(path)

    def test_larger_tol_merges_near_points(self):
        loop = np.array([[0, 0, 0], [0.5, 0, 0], [0.5001, 0, 0], [1, 0, 0]])

        default = SurfaceBase.sort_points([loop], far_point=[-10, 0, 0])
        coarse = SurfaceBase.sort_points([loop], far_point=[-10, 0, 0], tol=0.01)

        self.assertEqual(len(default), 4)  # 0.0001 apart survives the default tol
        self.assertEqual(len(coarse), 3)  # ...but is merged at tol=0.01


class PublicApiTests(unittest.TestCase):
    def test_exports(self):
        import classy_blocks as cb

        self.assertTrue(hasattr(cb, "SurfaceBase"))
        self.assertTrue(hasattr(cb, "TriangulatedSurface"))
