import os
import tempfile
import unittest

import numpy as np
import trimesh

from classy_blocks.construct.curves.interpolated import LinearInterpolatedCurve
from classy_blocks.construct.surfaces.revolved import RevolvedSurface
from classy_blocks.util import functions as f


def vertical_cylinder_surface():
    # a straight profile at radius 2, from z=0 to z=1, revolved about +z
    curve = LinearInterpolatedCurve([[2, 0, 0], [2, 0, 1]])
    return RevolvedSurface(curve, axis=[0, 0, 1])


class RevolvedSurfaceGetPointTests(unittest.TestCase):
    def setUp(self):
        self.surface = vertical_cylinder_surface()

    def test_get_point_at_zero_angle_is_the_curve(self):
        np.testing.assert_allclose(self.surface.get_point(0, 0), [2, 0, 0], atol=1e-9)

    def test_get_point_revolves_about_axis(self):
        np.testing.assert_allclose(self.surface.get_point(1, np.pi / 2), [0, 2, 1], atol=1e-9)

    def test_all_points_lie_on_the_cylinder(self):
        for param in (0.0, 0.5, 1.0):
            for angle in (-np.pi, -1.0, 0.0, 1.0, np.pi):
                point = self.surface.get_point(param, angle)
                self.assertAlmostEqual(float(np.hypot(point[0], point[1])), 2.0, places=9)

    def test_bounds_combine_curve_and_angle(self):
        self.assertEqual(self.surface.bounds, ((0, 1), (-np.pi, np.pi)))

    def test_axis_and_center(self):
        np.testing.assert_allclose(self.surface.axis, [0, 0, 1], atol=1e-9)
        np.testing.assert_allclose(self.surface.center, [0, 0, 0], atol=1e-9)


class RevolvedSurfaceTransformTests(unittest.TestCase):
    """Transformations of RevolvedSurface via ElementBase"""

    def setUp(self):
        self.surface = vertical_cylinder_surface()

    def test_translate(self):
        """Translate moves origin and axis tip, changing surface position"""
        displacement = [1, 2, 3]
        original_surface = self.surface
        translated_surface = self.surface.copy().translate(displacement)

        # Origin should be translated
        np.testing.assert_allclose(original_surface.center + displacement, translated_surface.center, atol=1e-9)

        # A point on the original surface should be translated by the same amount
        original_point = original_surface.get_point(0.5, 0)
        translated_point = translated_surface.get_point(0.5, 0)
        np.testing.assert_allclose(original_point + displacement, translated_point, atol=1e-9)

        # Axis direction should remain unchanged after translation
        np.testing.assert_allclose(translated_surface.axis, [0, 0, 1], atol=1e-9)

    def test_rotate(self):
        """Rotate changes the axis direction and curve position"""
        axis = [0, 0, 1]
        angle = np.pi / 2
        origin = [0, 0, 0]

        original_surface = self.surface
        rotated_surface = self.surface.copy().rotate(angle, axis, origin)

        # Axis direction should remain along z after rotation about z
        np.testing.assert_allclose(rotated_surface.axis, original_surface.axis, atol=1e-9)

        # Center should remain at origin for rotation about origin
        np.testing.assert_allclose(rotated_surface.center, original_surface.center, atol=1e-9)

        # A point at parameter 0 (on the curve profile at angle 0) should move
        # The point [2, 0, 0] should move to [0, 2, 0]
        # original_point = original_surface.get_point(0, 0)
        rotated_point = rotated_surface.get_point(0, 0)
        np.testing.assert_allclose(rotated_point, [0, 2, 0], atol=1e-9)

    def test_scale(self):
        """Scale changes the radius of the cylinder"""
        scale_factor = 2.0
        origin = [0, 0, 0]

        original_surface = self.surface
        scaled_surface = self.surface.copy().scale(scale_factor, origin)

        # Center remains at origin
        np.testing.assert_allclose(scaled_surface.center, original_surface.center, atol=1e-9)

        # Axis direction should not change
        np.testing.assert_allclose(scaled_surface.axis, original_surface.axis, atol=1e-9)

        # Points should be scaled: radius should double
        original_point = original_surface.get_point(0, 0)  # [2, 0, 0]
        scaled_point = scaled_surface.get_point(0, 0)  # should be [4, 0, 0]
        np.testing.assert_allclose(scaled_point, original_point * scale_factor, atol=1e-9)

    def test_scale_default_origin(self):
        """Scale with no origin given uses the surface center"""
        scale_factor = 0.5
        original_surface = self.surface
        scaled_surface = self.surface.copy().scale(scale_factor)

        # Center should not move (scaled about itself)
        np.testing.assert_allclose(scaled_surface.center, original_surface.center, atol=1e-9)

        # Radius should be halved
        original_point = original_surface.get_point(0, 0)  # [2, 0, 0]
        scaled_point = scaled_surface.get_point(0, 0)  # should be [1, 0, 0]
        np.testing.assert_allclose(scaled_point, original_point * scale_factor, atol=1e-9)

    def test_copy(self):
        """Copy creates an independent object"""
        copied_surface = self.surface.copy()

        # Initially same
        original_point = self.surface.get_point(0.5, 0)
        np.testing.assert_allclose(original_point, copied_surface.get_point(0.5, 0), atol=1e-9)

        # Transform original
        self.surface.translate([1, 1, 1])

        # Copy should not change - still at original position
        np.testing.assert_allclose(copied_surface.get_point(0.5, 0), original_point, atol=1e-9)

        # Original should be at new position
        np.testing.assert_allclose(self.surface.get_point(0.5, 0), original_point + f.vector(1, 1, 1), atol=1e-9)

    def test_chaining_transforms(self):
        """Multiple transforms can be chained"""
        transformed_surface = self.surface.copy().translate([1, 0, 0]).scale(2.0, [1, 0, 0])

        # After translate [1,0,0]: radius still 2, center at [1,0,0], curve point at [3,0,0]
        # After scale 2 about [1,0,0]: center stays at [1,0,0], curve point becomes [5,0,0]
        expected_point = [5, 0, 0]
        np.testing.assert_allclose(transformed_surface.get_point(0, 0), expected_point, atol=1e-9)


class RevolvedSurfaceClosestPointTests(unittest.TestCase):
    def setUp(self):
        # cylinder radius 2, spanning z in [-1, 1], full revolution
        curve = LinearInterpolatedCurve([[2, 0, -1], [2, 0, 1]])
        self.surface = RevolvedSurface(curve, axis=[0, 0, 1])

    def test_closest_along_reference_meridian(self):
        np.testing.assert_allclose(self.surface.get_closest_point([10, 0, 0]), [2, 0, 0], atol=1e-6)

    def test_closest_at_ninety_degrees(self):
        np.testing.assert_allclose(self.surface.get_closest_point([0, 10, 0]), [0, 2, 0], atol=1e-6)

    def test_closest_at_arbitrary_azimuth_and_height(self):
        # azimuth 45 degrees, height 0.3
        result = self.surface.get_closest_point([5, 5, 0.3])
        expected = [2 * np.cos(np.pi / 4), 2 * np.sin(np.pi / 4), 0.3]
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_distance_is_radial_gap(self):
        result = self.surface.get_closest_point([5, 0, 0])
        self.assertAlmostEqual(float(np.linalg.norm(result - np.array([5, 0, 0]))), 3.0, places=6)

    def test_sector_clamps_azimuth_to_boundary(self):
        curve = LinearInterpolatedCurve([[2, 0, -1], [2, 0, 1]])
        sector = RevolvedSurface(curve, axis=[0, 0, 1], angle_bounds=(-np.pi / 4, np.pi / 4))

        # query at +90 degrees is outside the +/-45 degree sector -> snaps to +45 boundary
        result = sector.get_closest_point([0, 5, 0])
        self.assertAlmostEqual(float(np.hypot(result[0], result[1])), 2.0, places=6)
        self.assertAlmostEqual(float(np.arctan2(result[1], result[0])), np.pi / 4, places=6)


class RevolvedSurfaceFromMeshTests(unittest.TestCase):
    def setUp(self):
        # cylinder radius 2, height 4, axis +z, centered at origin (z in [-2, 2])
        self.cylinder = trimesh.creation.cylinder(radius=2, height=4)

    def test_from_mesh_recovers_radius(self):
        surface = RevolvedSurface.from_mesh(self.cylinder, axis=[0, 0, 1], anchor=[1, 0, 0], origin=[0, 0, 0])
        closest = surface.get_closest_point([10, 0, 0])
        self.assertAlmostEqual(float(np.hypot(closest[0], closest[1])), 2.0, places=6)

    def test_from_file_roundtrip(self):
        path = os.path.join(tempfile.mkdtemp(), "cyl.stl")
        self.cylinder.export(path)

        surface = RevolvedSurface.from_file(path, axis=[0, 0, 1], anchor=[1, 0, 0], origin=[0, 0, 0])
        closest = surface.get_closest_point([10, 0, 0])
        self.assertAlmostEqual(float(np.hypot(closest[0], closest[1])), 2.0, places=6)

    def test_anchor_height_is_irrelevant(self):
        low = RevolvedSurface.from_mesh(self.cylinder, axis=[0, 0, 1], anchor=[1, 0, -1.5], origin=[0, 0, 0])
        high = RevolvedSurface.from_mesh(self.cylinder, axis=[0, 0, 1], anchor=[1, 0, 1.5], origin=[0, 0, 0])

        np.testing.assert_allclose(low.get_closest_point([10, 0, 0]), high.get_closest_point([10, 0, 0]), atol=1e-6)

    def test_on_axis_anchor_raises(self):
        with self.assertRaises(ValueError):
            RevolvedSurface.from_mesh(self.cylinder, axis=[0, 0, 1], anchor=[0, 0, 1], origin=[0, 0, 0])

    def test_cut_missing_the_mesh_raises(self):
        box = trimesh.creation.box(extents=[2, 2, 2])
        box.apply_translation([10, 10, 0])  # entirely off the y=0 meridional plane

        with self.assertRaises(ValueError):
            RevolvedSurface.from_mesh(box, axis=[0, 0, 1], anchor=[1, 0, 0], origin=[0, 0, 0])


class RevolvedSurfaceCoverageTests(unittest.TestCase):
    def setUp(self):
        # cylinder radius 2, height 4, axis +z, centered at origin (z in [-2, 2])
        self.cylinder = trimesh.creation.cylinder(radius=2, height=4)

    def test_get_cross_sections_not_exposed(self):
        """RevolvedSurface narrows the interface: no get_cross_sections (that's TriangulatedSurface's job)"""
        surface = RevolvedSurface(LinearInterpolatedCurve([[2, 0, 0], [2, 0, 1]]), axis=[0, 0, 1])
        self.assertFalse(hasattr(surface, "get_cross_sections"))

    def test_opposite_anchor_selects_opposite_meridian(self):
        # the revolved surfaces are identical by symmetry; check the extracted profile instead
        plus = RevolvedSurface.from_mesh(self.cylinder, axis=[0, 0, 1], anchor=[1, 0, 0], origin=[0, 0, 0])
        minus = RevolvedSurface.from_mesh(self.cylinder, axis=[0, 0, 1], anchor=[-1, 0, 0], origin=[0, 0, 0])

        self.assertGreaterEqual(plus.curve.discretize()[:, 0].min(), -1e-6)  # profile on +x half
        self.assertLessEqual(minus.curve.discretize()[:, 0].max(), 1e-6)  # profile on -x half

    def test_profile_runs_bottom_to_top(self):
        surface = RevolvedSurface.from_mesh(self.cylinder, axis=[0, 0, 1], anchor=[1, 0, 0], origin=[0, 0, 0])
        profile = surface.curve.discretize()
        self.assertLess(profile[0][2], profile[-1][2])  # starts near z=-2, ends near z=+2

    def test_empty_side_raises(self):
        box = trimesh.creation.box(extents=[2, 2, 2])
        box.apply_translation([-5, 0, 0])  # entirely on -x side, but still crosses the y=0 meridional plane

        with self.assertRaises(ValueError):
            RevolvedSurface.from_mesh(box, axis=[0, 0, 1], anchor=[1, 0, 0], origin=[0, 0, 0])


class RevolvedSurfaceExportTests(unittest.TestCase):
    """Test that RevolvedSurface is properly exported from the package root"""

    def test_revolved_surface_exported_from_package_root(self):
        """RevolvedSurface should be importable from classy_blocks package root"""
        import classy_blocks

        self.assertTrue(hasattr(classy_blocks, "RevolvedSurface"))
        self.assertIs(classy_blocks.RevolvedSurface, RevolvedSurface)
