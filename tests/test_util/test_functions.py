import unittest

import numpy as np

from classy_blocks.util import functions as f


class TestFunctions(unittest.TestCase):
    def assert_np_equal(self, a, b, msg=None):
        return np.testing.assert_array_equal(a, b, err_msg=msg)

    def assert_np_almost_equal(self, a, b, msg=None):
        return np.testing.assert_array_almost_equal(a, b, err_msg=msg)

    def test_deg2rad(self):
        """degrees to radians"""
        deg = 30
        self.assertEqual(deg * np.pi / 180, f.deg2rad(deg))

    def test_rad2deg(self):
        """radians to degrees"""
        rad = np.pi / 3
        self.assertEqual(rad * 180 / np.pi, f.rad2deg(rad))

    def test_unit_vector(self):
        """scale vector to magnitude 1"""
        vector = f.vector(3, 5, 7)
        unit_vector = vector / np.linalg.norm(vector)

        self.assert_np_equal(unit_vector, f.unit_vector(vector))

    def test_norm_vector(self):
        """Vector norm"""
        self.assertEqual(f.norm(f.vector(1, 0, 0)), 1)

    def test_angle_between(self):
        """angle between two vectors"""
        v1 = f.vector(0, 0, 1)
        v2 = f.vector(0, 2, 0)

        self.assertEqual(f.angle_between(v1, v2), np.pi / 2)
        self.assertEqual(f.angle_between(v1, v1), 0)

        v1 = f.vector(1, 0, 0)
        v2 = f.vector(1, 1, 0)
        self.assertAlmostEqual(f.angle_between(v1, v2), np.pi / 4)

    def test_arbitrary_rotation_point(self):
        """rotation of a point from another origin"""
        point = f.vector(0, 2, 0)
        origin = f.vector(0, 1, 0)
        axis = f.vector(0, 0, 1)

        self.assert_np_equal(f.rotate(point, -np.pi / 2, axis, origin), f.vector(1, 1, 0))

    def test_arbitrary_rotation_axis(self):
        """rotation of a point around arbitrary axis"""
        point = f.vector(1, 0, 0)
        origin = f.vector(0, 0, 0)
        axis = f.vector(1, 1, 0)

        self.assert_np_almost_equal(f.rotate(point, np.pi, axis, origin), f.vector(0, 1, 0))

    def test_to_polar_z_axis(self):
        """cartesian coordinate system to polar c.s., rotation around z-axis"""
        cartesian = f.vector(2, 2, 5)
        polar = f.vector(8**0.5, np.pi / 4, 5)

        self.assert_np_almost_equal(polar, f.to_polar(cartesian, axis="z"))

    def test_to_polar_x_axis(self):
        """cartesian coordinate system to polar c.s., rotation around x-axis"""
        cartesian = f.vector(5, 2, 2)
        polar = f.vector(8**0.5, np.pi / 4, 5)

        self.assert_np_almost_equal(polar, f.to_polar(cartesian, axis="x"))

    def test_to_polar_invalid_axis(self):
        with self.assertRaises(ValueError):
            _ = f.to_polar([0, 1, 1], "y")

    def test_lin_map(self):
        """map a value"""
        self.assertEqual(f.lin_map(10, 0, 100, 0, 1000), 100)

    def test_lin_map_limit(self):
        """map a value within given limits"""
        self.assertEqual(f.lin_map(200, 0, 10, 0, 100, limit=True), 100)
        self.assertEqual(f.lin_map(-5, 0, 10, 0, 100, limit=True), 0)

    def test_to_cartesian_point(self):
        """polar point to xyz"""
        # polar point
        p = np.array([1, np.pi / 2, 2])

        # cartesian versions
        # axis: z
        self.assert_np_almost_equal(f.to_cartesian(p, axis="z"), f.vector(0, 1, 2))

        self.assert_np_almost_equal(f.to_cartesian(p, axis="x"), f.vector(2, 0, 1))

        self.assert_np_almost_equal(f.to_cartesian(p, axis="x", direction=-1), f.vector(2, 0, -1))

    def test_polar2cartesian_x_axis(self):
        cp = np.array([32, 198, 95])
        pp = f.to_polar(cp, axis="x")

        self.assert_np_almost_equal(cp, f.to_cartesian(pp, axis="x"))

    def test_polar2cartesian_z_axis(self):
        cp = np.array([32, 198, 95])
        pp = f.to_polar(cp, axis="z")

        self.assert_np_almost_equal(cp, f.to_cartesian(pp, axis="z"))

    def test_to_cartesian_asserts(self):
        """garbage input to f.to_cartesian()"""
        p = np.array([1, 0, 1])

        with self.assertRaises(ValueError):
            f.to_cartesian(p, direction=0)

        with self.assertRaises(ValueError):
            f.to_cartesian(p, axis="a")

    def test_arc_length_3point_half(self):
        a = f.vector(0, 0, 0)
        b = f.vector(1, 1, 0)
        c = f.vector(2, 0, 0)
        self.assertAlmostEqual(f.arc_length_3point(a, b, c), np.pi)

    def test_arc_length_3point_quarter(self):
        a = f.vector(0, 0, 0)
        s2 = 2**0.5 / 2
        b = f.vector(1 - s2, s2, 0)
        c = f.vector(1, 1, 0)
        self.assertAlmostEqual(f.arc_length_3point(a, b, c), np.pi / 2)

    def test_arc_length_3point_3quarter(self):
        a = f.vector(0, 0, 0)
        s2 = 2**0.5 / 2
        b = f.vector(1 + s2, s2, 0)
        c = f.vector(1, -1, 0)
        self.assertAlmostEqual(f.arc_length_3point(a, b, c), 3 * np.pi / 2)

    def test_arc_length_3point_full(self):
        a = f.vector(0, 0, 0)
        b = f.vector(2, 0, 0)
        c = f.vector(0, 0, 0)

        with self.assertRaises(ValueError):
            self.assertAlmostEqual(f.arc_length_3point(a, b, c), 2 * np.pi)

    def test_arc_length_3point_zero(self):
        a = f.vector(0, 0, 0)
        b = f.vector(0, 0, 0)
        c = f.vector(0, 0, 0)

        with self.assertRaises(ValueError):
            self.assertAlmostEqual(f.arc_length_3point(a, b, c), 0)

    def test_mirror_yz(self):
        point = [2, 0, 0]
        mirrored = f.mirror(point, [1, 0, 0], [0, 0, 0])

        np.testing.assert_almost_equal(mirrored, [-2, 0, 0])

    def test_mirror_yz_origin(self):
        point = [2, 0, 0]
        mirrored = f.mirror(point, [1, 0, 0], [1, 1, 1])

        np.testing.assert_almost_equal(mirrored, [0, 0, 0])

    def test_mirror_xz(self):
        point = [0, 2, 0]
        mirrored = f.mirror(point, [0, 1, 0], [0, 0, 0])

        np.testing.assert_almost_equal(mirrored, [0, -2, 0])

    def test_mirror_xz_origin(self):
        point = [0, 2, 0]
        mirrored = f.mirror(point, [0, 1, 0], [0, 1, 0])

        np.testing.assert_almost_equal(mirrored, [0, 0, 0])

    def test_mirror_xy(self):
        point = [0, 0, 2]
        mirrored = f.mirror(point, [0, 0, 1], [0, 0, 0])

        np.testing.assert_almost_equal(mirrored, [0, 0, -2])

    def test_mirror_xy_origin(self):
        point = [0, 0, 2]
        mirrored = f.mirror(point, [0, 0, 1], [0, 0, 1])

        np.testing.assert_almost_equal(mirrored, [0, 0, 0])

    def test_mirror_arbitrary(self):
        point = [2, 2, 2]
        mirrored = f.mirror(point, [1, 1, 1], [0, 0, 0])

        np.testing.assert_almost_equal(mirrored, [-2, -2, -2])

    def test_mirror_arbitrary_origin(self):
        point = [2, 2, 2]
        mirrored = f.mirror(point, [1, 1, 1], [1, 1, 1])

        np.testing.assert_almost_equal(mirrored, [0, 0, 0])
