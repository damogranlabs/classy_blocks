import unittest

import numpy as np

from classy_blocks.base.exceptions import CylinderCreationError, FrustumCreationError
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.construct.shapes.elbow import Elbow
from classy_blocks.construct.shapes.frustum import Frustum
from classy_blocks.construct.shapes.rings import ExtrudedRing, RevolvedRing
from classy_blocks.construct.shapes.sphere import EighthSphere, Hemisphere
from classy_blocks.util import functions as f


class ShapeTests(unittest.TestCase):
    """Common shape methods and properties"""

    def setUp(self):
        self.cylinder = Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])

    def test_translate(self):
        """Translate a cylinder (uses the 'parts' property)"""
        self.cylinder.translate([1, 0, 0])

        np.testing.assert_array_equal(self.cylinder.sketch_1.center, [1, 0, 0])

    def test_cylinder_center(self):
        """Center of a cylinder"""
        np.testing.assert_almost_equal(self.cylinder.center, [0.5, 0, 0])

    def test_set_start_patch(self):
        """Start patch on all operations"""
        self.cylinder.set_start_patch("test")

        for operation in self.cylinder.operations:
            self.assertEqual(operation.bottom_face.patch_name, "test")

    def test_set_end_patch(self):
        """Start patch on all operations"""
        self.cylinder.set_end_patch("test")

        for operation in self.cylinder.operations:
            self.assertEqual(operation.top_face.patch_name, "test")

    def test_outer_patch(self):
        """Start patch on all operations"""
        self.cylinder.set_outer_patch("test")

        for operation in self.cylinder.shell:
            self.assertEqual(operation.patch_names[self.cylinder.outer_patch], "test")

    def test_inner_patch_extruded(self):
        """Inner patch on an extruded ring"""
        ring = ExtrudedRing([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.4)
        ring.set_inner_patch("test")

        for operation in ring.shell:
            self.assertEqual(operation.patch_names["left"], "test")

    def test_inner_patch_revolved(self):
        face = Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        face.translate([0, 1, 0])
        ring = RevolvedRing([0, 0, 0], [1, 0, 0], face)

        ring.set_inner_patch("test")

        for operation in ring.shell:
            self.assertEqual(operation.patch_names["front"], "test")


class ElbowTests(unittest.TestCase):
    """Tests of the Elbow shape"""

    def setUp(self):
        self.center_point_1 = f.vector(1.0, 1.0, 1.0)
        self.radius_point_1 = f.vector(2.0, 1.0, 1.0)
        self.normal_1 = f.vector(0, 1, 0)

        self.sweep_angle = -np.pi / 3

        self.arc_center = f.vector(3.0, 1.0, 1.0)
        self.rotation_axis = f.vector(1.0, 1.0, 2.0)

        self.radius_2 = 0.4

    @property
    def radius_1(self) -> float:
        """Start radius"""
        return f.norm(self.radius_point_1 - self.center_point_1)

    @property
    def elbow(self) -> Elbow:
        """The test subject"""
        return Elbow(
            self.center_point_1,
            self.radius_point_1,
            self.normal_1,
            self.sweep_angle,
            self.arc_center,
            self.rotation_axis,
            self.radius_2,
        )

    def test_radius_1(self):
        """Radius of the start sketch"""
        self.assertAlmostEqual(self.elbow.sketch_1.radius, self.radius_1)

    def test_radius_2(self):
        """Radius of the end sketch"""
        self.assertAlmostEqual(self.elbow.sketch_2.radius, self.radius_2)

    def test_radius_mid_nonuniform(self):
        """Radius of the middle sketch"""
        # should be between 1 and 2
        self.assertAlmostEqual(self.elbow.sketch_mid.radius, (self.radius_1 + self.radius_2) / 2)

    def test_radius_mid_uniform(self):
        """Radius of the middle sketch with a uniform cross-section"""
        # should be the same as 1 and 2
        self.radius_2 = f.norm(self.center_point_1 - self.radius_point_1)
        self.assertAlmostEqual(self.elbow.sketch_mid.radius, self.radius_2)

    def test_sketch_positions(self):
        """Sketch positions after Elbow transforms"""
        elbow = self.elbow

        center_1 = elbow.sketch_1.center
        center_2 = f.rotate(center_1, self.sweep_angle, self.rotation_axis, self.arc_center)

        np.testing.assert_array_almost_equal(elbow.sketch_2.center, center_2)


class RevolvedRingTests(unittest.TestCase):
    """RevolvedRing creation and manipulation"""

    def setUp(self):
        self.face = Face([[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]])
        self.axis_point_1 = [0, 0, 0]
        self.axis_point_2 = [1, 0, 0]
        self.n_segments = 8

        self.ring = RevolvedRing(self.axis_point_1, self.axis_point_2, self.face, self.n_segments)

    def test_set_inner_patch(self):
        """Inner faces of the ring"""
        self.ring.set_inner_patch("inner")

        for operation in self.ring.operations:
            self.assertEqual(operation.patch_names["front"], "inner")

    def test_set_outer_patch(self):
        """Outer faces of the ring"""
        self.ring.set_outer_patch("outer")

        for operation in self.ring.operations:
            self.assertEqual(operation.patch_names["back"], "outer")

    def test_chop_ring_tangential(self):
        self.ring.chop_tangential(count=10)
        self.ring.chop_radial(count=1)
        self.ring.chop_axial(count=1)

        for operation in self.ring.operations:
            self.assertEqual(operation.chops[self.ring.tangential_axis][0].count, 10)


class SphereTests(unittest.TestCase):
    def test_sphere_radii(self):
        """Check that all radii, defined by outer points, are as specified"""
        center = f.vector(1, 1, 1)
        sphere = Hemisphere(center, [2, 2, 1], [0, 0, 1])
        radius = 2**0.5

        for loft in sphere.shell:
            for point in loft.get_face("right").point_array:
                self.assertAlmostEqual(f.norm(point - center), radius)

    def test_start_patch(self):
        sphere = Hemisphere([0, 0, 0], [1, 0, 0], [0, 0, 1])
        sphere.set_start_patch("flat")

        n_patches = 0

        for operation in sphere.operations:
            if len(operation.patch_names) > 0:
                n_patches += 1

        self.assertEqual(n_patches, 12)

    def test_core(self):
        sphere = EighthSphere([0, 0, 0], [1, 0, 0], [0, 0, 1])

        self.assertEqual(len(sphere.core), 1)


class FrustumTests(unittest.TestCase):
    def test_non_perpendicular_axis_radius(self):
        with self.assertRaises(FrustumCreationError):
            Frustum([0, 0, 0], [1, 1, 0], [0, 1, 0], 0.4)

    def test_curved_side(self):
        """Create a Frustum with curved side edges"""
        frustum = Frustum([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.4, 0.333)

        for operation in frustum.shell:
            edges = operation.edges

            self.assertEqual(edges[1][5].kind, "arc")
            self.assertEqual(edges[2][6].kind, "arc")


class CylinderTests(unittest.TestCase):
    def setUp(self):
        self.axis_point_1 = [0.0, 0.0, 0.0]
        self.axis_point_2 = [1.0, 0.0, 0.0]
        self.radius_point_1 = [0.0, 1.0, 0.0]
        self.cylinder = Cylinder(self.axis_point_1, self.axis_point_2, self.radius_point_1)

    def test_non_perpendicular_axis_radius(self):
        with self.assertRaises(CylinderCreationError):
            Cylinder([0, 0, 0], [1, 0, 0], [1, 0, 0])

    def test_edges(self):
        """Bug check: check that all edges are translated equally"""
        for face in self.cylinder.sketch_2.shell:
            # in Disk, 2nd edge of shell's face is Origin
            self.assertEqual(face.edges[1].origin.position[0], 1)

    def test_core(self):
        """Make sure cylinder's core is represented correctly"""
        self.assertEqual(len(self.cylinder.core), 4)

    def test_chop_radial_start_size(self):
        """Radial chop and start_size corrections"""
        self.cylinder.chop_radial(start_size=0.1)

        self.assertNotEqual(self.cylinder.shell[0].chops[0][0].start_size, 0.1)

    def test_chop_radial_end_size(self):
        """Radial chop and end_size corrections"""
        self.cylinder.chop_radial(end_size=0.1)

        self.assertNotEqual(self.cylinder.shell[0].chops[0][0].end_size, 0.1)
