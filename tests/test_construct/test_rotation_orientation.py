import unittest
import warnings

import numpy as np
from parameterized import parameterized

import classy_blocks as cb
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.shapes.elbow import Elbow
from classy_blocks.construct.shapes.rings import RevolvedRing
from classy_blocks.construct.stack import RevolvedStack
from classy_blocks.mesh import Mesh
from classy_blocks.util import functions as f

# hex cell-model faces from OpenFOAM's etc/cellModels
HEX_FACES = ((0, 4, 7, 3), (1, 2, 6, 5), (0, 1, 5, 4), (3, 7, 6, 2), (0, 3, 2, 1), (4, 5, 6, 7))


def face_area_vector(points):
    """Area vector of a quad, fan from its first point (OpenFOAM rule)"""
    p0 = points[0]
    return 0.5 * sum(np.cross(points[i] - p0, points[i + 1] - p0) for i in (1, 2))


def block_volumes(mesh):
    """Signed volumes of all blocks; negative volume = inside-out block"""
    mesh.assemble()

    volumes = []

    for block in mesh.blocks:
        points = np.array([v.position for v in block.vertices])
        volume = (
            sum(np.dot(points[list(face)].mean(axis=0), face_area_vector(points[list(face)])) for face in HEX_FACES)
            / 3.0
        )
        volumes.append(volume)

    return volumes


class RotationOrientationTests(unittest.TestCase):
    def test_revolve_angle_directions(self):
        """Blocks of a Revolve must be outward for both rotation directions"""
        for angle in (np.pi / 3, -np.pi / 3):
            mesh = Mesh()
            base = Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [cb.Arc([0.5, -0.2, 0]), None, None, None])
            revolve = cb.Revolve(base, angle, [0, -1, 0], [-2, 0, 0])
            revolve.chop(0, count=2)
            revolve.chop(1, count=2)
            revolve.chop(2, count=4)
            mesh.add(revolve)

            self.assertTrue(all(volume > 0 for volume in block_volumes(mesh)))

    @parameterized.expand(
        [
            (np.pi / 3, 0.4),
            (-np.pi / 3, 0.4),
            (np.pi / 3, 1.5),
            (-np.pi / 3, 1.5),
        ]
    )
    def test_elbow_sweep_directions(self, sweep, radius_2):
        """All blocks of an Elbow must be outward for both sweep directions"""
        elbow = Elbow([0, 0, 0], [1, 0, 0], [0, 1, 0], sweep, [2, 0, 0], [0, 0, 1], radius_2)
        elbow.chop_tangential(count=4)
        elbow.chop_radial(count=2)
        elbow.chop_axial(count=2)

        mesh = Mesh()
        mesh.add(elbow)

        self.assertTrue(all(volume > 0 for volume in block_volumes(mesh)))

    def test_revolved_ring_and_stack(self):
        """Revolved shapes built on LoftedShape must be outward as well"""
        ring = RevolvedRing([0, 0, 0], [1, 0, 0], Face([[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]]), 8)

        mesh = Mesh()
        mesh.add(ring)
        self.assertTrue(all(volume > 0 for volume in block_volumes(mesh)))

        stack = RevolvedStack(cb.OneCoreDisk([0, 0, 0], [1, 0, 0], [0, 0, 1]), np.pi / 6, [0, 1, 0], [2, 0, 0], 4)

        mesh = Mesh()
        mesh.add(stack)
        self.assertTrue(all(volume > 0 for volume in block_volumes(mesh)))

    def test_translation_shapes_stay_outward(self):
        """Translation-based shapes must not be touched by the correction"""
        for solid in (cb.Box([0, 0, 0], [1, 1, 1]), cb.Cylinder([0, 0, 0], [0, 0, 1], [1, 0, 0])):
            mesh = Mesh()
            mesh.add(solid)
            self.assertTrue(all(volume > 0 for volume in block_volumes(mesh)))

    def test_revolve_side_arc_direction(self):
        """The arc of a negative-angle Revolve sweeps the same sector as its positive twin"""
        mesh = Mesh()
        base = Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [cb.Arc([0.5, -0.2, 0]), None, None, None])
        revolve = cb.Revolve(base, -np.pi / 3, [0, -1, 0], [-2, 0, 0])
        revolve.chop(0, count=1)
        revolve.chop(1, count=1)
        revolve.chop(2, count=1)
        mesh.add(revolve)
        mesh.assemble()

        for wire in mesh.blocks[0].wires.get_axis_beams(2):
            if set(wire.corners) == {0, 4}:
                third_point = wire.edge.third_point.position
                break
        else:
            self.fail("side edge not found")

        expected = f.rotate(np.array([0, 0, 0]), -np.pi / 6, [0, -1, 0], [-2, 0, 0])
        np.testing.assert_array_almost_equal(third_point, expected)

    def test_elbow_start_end_patches(self):
        """Start/end patches stay on the start/end faces for both sweep directions"""
        for sweep in (np.pi / 3, -np.pi / 3):
            elbow = Elbow([0, 0, 0], [1, 0, 0], [0, 1, 0], sweep, [2, 0, 0], [0, 0, 1], 0.4)
            elbow.set_start_patch("inlet")
            elbow.set_end_patch("outlet")

            self.assertEqual(elbow.sketch_1.faces[0].patch_name, "inlet")
            self.assertEqual(elbow.sketch_2.faces[0].patch_name, "outlet")

    def test_auto_flip_warns(self):
        """An inverting sweep warns about the auto-flip; a non-inverting one doesn't"""
        # matches test_elbow_sweep_directions' inverting case: positive sweep
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Elbow([0, 0, 0], [1, 0, 0], [0, 1, 0], np.pi / 3, [2, 0, 0], [0, 0, 1], 0.4)
        self.assertTrue(any("inside-out" in str(w.message) for w in caught))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Elbow([0, 0, 0], [1, 0, 0], [0, 1, 0], -np.pi / 3, [2, 0, 0], [0, 0, 1], 0.4)
        self.assertFalse(any("inside-out" in str(w.message) for w in caught))
