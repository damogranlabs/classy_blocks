import unittest

import numpy as np

from classy_blocks.base.exceptions import (
    CylinderCreationError,
    ElbowCreationError,
    ExtrudedRingCreationError,
    FrustumCreationError,
)
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.construct.shapes.elbow import Elbow
from classy_blocks.construct.shapes.frustum import Frustum
from classy_blocks.construct.shapes.rings import ExtrudedRing
from classy_blocks.mesh import Mesh
from classy_blocks.util.constants import TOL


class ElbowChainingTests(unittest.TestCase):
    """Chaining of elbow to everything elbow-chainable"""

    def setUp(self):
        self.elbow = Elbow(
            [0, 0, 0],  # center_point_1
            [1, 0, 0],  # radius_point_1
            [0, 1, 0],  # normal_1
            -np.pi / 2,  # sweep angle
            [2, 0, 0],  # arc_center
            [0, 0, 1],  # rotation_axis
            1.0,  # radius_2
        )

        self.mesh = Mesh()

    def check_success(self, chained_shape, end_center):
        """adds the chained stuff to mesh and
        checks the number of vertices as a measurement of success"""
        self.mesh.add(self.elbow)
        self.mesh.add(chained_shape)

        self.mesh.assemble()

        self.assertEqual(len(self.mesh.block_list.blocks), 24)
        self.assertEqual(len(self.mesh.vertex_list.vertices), 3 * 17)

        np.testing.assert_allclose(chained_shape.sketch_2.center, end_center)

    def test_to_elbow_end(self):
        """Chain an elbow to an elbow on an end sketch"""
        chained = Elbow.chain(
            self.elbow,  # source
            -np.pi / 2,  # sweep_angle
            [2, 0, 0],  # arc_center
            [0, 0, 1],  # rotation_axis
            1,  # radius_2
            False,
        )  # start_face

        self.check_success(chained, [4, 0, 0])

    def test_chain_on_invalid_start_face(self):
        with self.assertRaises(ElbowCreationError):
            elbow = self.elbow.copy()
            # set invalid base shape for chaining
            elbow.sketch_1 = Face([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
            Elbow.chain(elbow, np.pi / 2, [2, 0, 0], [0, 0, 1], 1, True)

    def test_to_elbow_start(self):
        """Chain an elbow to an elbow on a start sketch"""
        chained = Elbow.chain(self.elbow, np.pi / 2, [2, 0, 0], [0, 0, 1], 1, True)
        self.check_success(chained, [2, -2, 0])

    def test_to_cylinder_start(self):
        """Chain an elbow to a cylinder on an end sketch"""
        chained = Cylinder.chain(self.elbow, 1)
        self.check_success(chained, [3, 2, 0])

    def test_to_cylinder_end(self):
        """Chain an elbow to a cylinder on a start sketch"""
        chained = Cylinder.chain(self.elbow, 1, start_face=True)
        self.check_success(chained, [0, -1, 0])

    def test_chain_frustum_invalid_length(self):
        with self.assertRaises(FrustumCreationError):
            Frustum.chain(self.elbow, -1, 0.5)

    def test_to_frustum_start(self):
        """Chain an elbow to a frustum on end sketch"""
        chained = Frustum.chain(self.elbow, 1, 0.5)
        self.check_success(chained, [3, 2, 0])

    def test_to_frustum_end(self):
        """Chain an elbow to a frustum on start sketch"""
        chained = Frustum.chain(self.elbow, 1, 0.5, start_face=True)
        self.check_success(chained, [0, -1, 0])


class RingChainingTests(unittest.TestCase):
    """Chaining of extruded rings"""

    def setUp(self):
        self.ring = ExtrudedRing(
            [0, 0, 0],  # center_point_1
            [1, 0, 0],  # center_point_2
            [0, 1, 0],  # radius_point_1
            0.8,  # inner_radius
            7,  # n_segments: deliberately use non-default
        )

        self.mesh = Mesh()

    def check_success(self, chained_shape, end_center):
        """adds the chained stuff to mesh and
        checks the number of vertices as a measurement of success"""
        self.mesh.add(self.ring)
        self.mesh.add(chained_shape)

        self.mesh.assemble()

        self.assertEqual(len(self.mesh.block_list.blocks), 2 * self.ring.sketch_1.n_segments)
        self.assertEqual(len(self.mesh.vertex_list.vertices), 3 * 2 * self.ring.sketch_1.n_segments)

        np.testing.assert_allclose(chained_shape.sketch_2.center, end_center, atol=TOL)

    def test_chain_invalid_length(self):
        with self.assertRaises(ExtrudedRingCreationError):
            ExtrudedRing.chain(self.ring, -1)

    def test_chain_end(self):
        """Chain an extruded ring on end face"""
        chained = ExtrudedRing.chain(self.ring, 1)
        self.check_success(chained, [2, 0, 0])

    def test_chain_start(self):
        """Chain an extruded ring on start face"""
        chained = ExtrudedRing.chain(self.ring, 1, start_face=True)
        self.check_success(chained, [-1, 0, 0])


class ExpandContractTests(unittest.TestCase):
    """Tests of shapes and their methods expand/contract/fill"""

    def setUp(self):
        self.mesh = Mesh()

    def check_success(self, shape_1, shape_2, n_blocks, n_vertices):
        self.mesh.add(shape_1)
        self.mesh.add(shape_2)
        self.mesh.assemble()

        self.assertEqual(len(self.mesh.block_list.blocks), n_blocks)
        self.assertEqual(len(self.mesh.vertex_list.vertices), n_vertices)

    def test_expand_cylinder(self):
        """Expand a ring from a cylinder"""
        cylinder = Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])
        expanded = ExtrudedRing.expand(cylinder, 0.25)

        self.check_success(cylinder, expanded, 20, 2 * (17 + 8))

    def test_expand_ring(self):
        """Expand a ring from a ring"""
        ring = ExtrudedRing([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.8, 9)
        expanded = ExtrudedRing.expand(ring, 0.25)

        self.check_success(ring, expanded, 18, 3 * 9 * 2)

    def test_contract_ring_invalid_radius(self):
        with self.assertRaises(ExtrudedRingCreationError):
            ring = ExtrudedRing([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.6, 9)
            _ = ExtrudedRing.contract(ring, 2)

    def test_contract_ring(self):
        """Contract a ring from another ring"""
        ring = ExtrudedRing([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.6, 9)
        contracted = ExtrudedRing.contract(ring, 0.2)
        self.check_success(ring, contracted, 18, 2 * 3 * 9)

    def test_fill_assert(self):
        """Make sure the source ring is made from 8 segments"""
        with self.assertRaises(CylinderCreationError):
            Cylinder.fill(ExtrudedRing([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.4, 9))

    def test_fill(self):
        """Fill an ExtrudedRing with a Cylinder"""
        ring = ExtrudedRing([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.4)
        fill = Cylinder.fill(ring)

        self.check_success(ring, fill, 20, 2 * (17 + 8))
