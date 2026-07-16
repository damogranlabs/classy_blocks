import os
import tempfile
import unittest

import numpy as np
import trimesh

from classy_blocks.construct.surfaces.triangulated import TriangulatedSurface
from classy_blocks.util import functions as f


class TriangulatedSurfaceClosestPointTests(unittest.TestCase):
    def setUp(self):
        self.radius = 2.0
        self.sphere = TriangulatedSurface(trimesh.creation.icosphere(radius=self.radius))
        # box centered at origin, faces at +/-1
        self.box = TriangulatedSurface(trimesh.creation.box(extents=[2, 2, 2]))

    def test_closest_point_on_box_face(self):
        result = self.box.get_closest_point([5, 0, 0])

        np.testing.assert_allclose(result, [1, 0, 0], atol=1e-9)

    def test_closest_point_on_sphere_lies_on_radius(self):
        result = self.sphere.get_closest_point([10, 0, 0])

        # icosphere is faceted, so allow a small tolerance
        self.assertAlmostEqual(f.norm(result), self.radius, places=1)

    def test_from_file_roundtrip(self):
        mesh = trimesh.creation.box(extents=[2, 2, 2])
        path = os.path.join(tempfile.mkdtemp(), "box.stl")
        mesh.export(path)

        surface = TriangulatedSurface.from_file(path)

        np.testing.assert_allclose(surface.get_closest_point([5, 0, 0]), [1, 0, 0], atol=1e-6)


class CrossSectionTests(unittest.TestCase):
    def setUp(self):
        self.box = TriangulatedSurface(trimesh.creation.box(extents=[2, 2, 2]))

        box_1 = trimesh.creation.box(extents=[2, 2, 2])
        box_2 = trimesh.creation.box(extents=[2, 2, 2])
        box_2.apply_translation([5, 0, 0])
        self.two_boxes = TriangulatedSurface(trimesh.util.concatenate([box_1, box_2]))

    def test_single_loop_from_box(self):
        loops = self.box.get_cross_sections([0, 0, 0], [0, 0, 1])

        self.assertEqual(len(loops), 1)
        # the loop is a square of side 2, lying in the z=0 plane
        loop = loops[0]
        np.testing.assert_allclose(loop[:, 2], 0, atol=1e-9)
        self.assertAlmostEqual(loop[:, 0].max(), 1.0, places=6)
        self.assertAlmostEqual(loop[:, 0].min(), -1.0, places=6)

    def test_two_loops_from_two_boxes(self):
        loops = self.two_boxes.get_cross_sections([0, 0, 0], [0, 0, 1])

        self.assertEqual(len(loops), 2)

    def test_missing_plane_returns_empty(self):
        loops = self.box.get_cross_sections([0, 0, 100], [0, 0, 1])

        self.assertEqual(loops, [])
