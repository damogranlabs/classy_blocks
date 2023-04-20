import unittest
import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.shapes.elbow import Elbow
from classy_blocks.construct.shapes.rings import RevolvedRing
from classy_blocks.construct.shapes.sphere import Hemisphere
from classy_blocks.construct.shapes.frustum import Frustum
from classy_blocks.util import functions as f


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

    def test_radius_mid(self):
        """Radius of the middle sketch"""
        # should be between 1 and 2
        self.assertAlmostEqual(self.elbow.sketch_mid.radius, (self.radius_1 + self.radius_2) / 2)


class RevolvedRingTests(unittest.TestCase):
    """RevolvedRing creation and manipulation"""

    def setUp(self):
        self.face = Face([[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]])
        self.axis_point_1 = [0, 0, 0]
        self.axis_point_2 = [1, 0, 0]
        self.n_segments = 8

    @property
    def ring(self) -> RevolvedRing:
        """The test subject"""
        return RevolvedRing(self.axis_point_1, self.axis_point_2, self.face, self.n_segments)

    def test_create(self):
        """Create a revolved ring"""
        _ = self.ring

    def test_set_inner_patch(self):
        """Inner faces of the ring"""
        ring = self.ring
        ring.set_inner_patch("inner")

        for operation in ring.operations:
            self.assertEqual(operation.patch_names["front"], "inner")


class SphereTests(unittest.TestCase):
    def test_sphere_radii(self):
        """Check that all radii, defined by outer points, are as specified"""
        center = f.vector(1, 1, 1)
        sphere = Hemisphere(center, [2, 2, 1], [0, 0, 1])
        radius = 2**0.5

        for loft in sphere.shell:
            for point in loft.get_face("right").point_array:
                self.assertAlmostEqual(f.norm(point - center), radius)


class FrustumTests(unittest.TestCase):
    def test_curved_side(self):
        """Create a Frustum with curved side edges"""
        frustum = Frustum([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.4, 0.333)

        for operation in frustum.shell:
            edges = operation.edges

            self.assertEqual(edges[1][5].kind, "arc")
            self.assertEqual(edges[2][6].kind, "arc")
