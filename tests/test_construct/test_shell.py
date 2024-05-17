from typing import List

import numpy as np
from parameterized import parameterized

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.box import Box
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shapes.shell import (
    AwareFace,
    AwareFaceStore,
    DisconnectedChopError,
    PointNotCoincidentError,
    SharedPoint,
    SharedPointNotFoundError,
    SharedPointStore,
    Shell,
)
from classy_blocks.types import OrientType
from classy_blocks.util import functions as f
from tests.fixtures.block import DataTestCase


class ShellTestsBase(DataTestCase):
    def setUp(self):
        super().setUp()

        data = self.get_single_data(0)

        self.bottom_face = Face(data.points[:4])
        self.top_face = Face(data.points[4:])

        self.loft = Loft(self.bottom_face, self.top_face)

    def get_face(self, orient) -> Face:
        return self.loft.get_face(orient)

    def get_point(self, orient, index):
        return self.loft.get_face(orient).points[index]


class SharedPointTests(ShellTestsBase):
    def get_shared_point(self, orient: OrientType, index: int):
        face = self.get_face(orient)
        point = face.points[index]
        sp = SharedPoint(point)
        sp.add(face, index)

        return sp

    def test_equal(self):
        face1 = self.bottom_face
        bp1 = SharedPoint(face1.points[1])
        bp1.add(face1, 1)

        face2 = self.loft.get_face("right")
        bp2 = SharedPoint(face2.points[1])
        bp2.add(face2, 1)

        self.assertEqual(bp1, bp2)

    def test_add(self):
        """Add a legit face/index to a BoundPoint"""
        point = self.get_point("bottom", 0)
        bp = SharedPoint(point)

        bp.add(self.bottom_face, 0)

        self.assertEqual(len(bp.faces), 1)

    def test_duplicated(self):
        """Do not add duplicate faces"""
        face = self.get_face("bottom")
        point = face.points[0]

        bp = SharedPoint(point)

        bp.add(face, 0)
        bp.add(face, 0)

        self.assertEqual(len(bp.faces), 1)

    def test_noncoincident(self):
        """Raise an exception when trying to add a bound point at a different location"""
        bp = SharedPoint(self.get_point("bottom", 0))

        with self.assertRaises(PointNotCoincidentError):
            bp.add(self.bottom_face, 1)

    def test_normal_single(self):
        """Normal when a single face is present in a bound point"""
        point = self.get_point("bottom", 0)
        bp = SharedPoint(point)
        bp.add(self.bottom_face, 0)

        np.testing.assert_array_almost_equal(bp.normal, self.bottom_face.normal)

    def test_normal_multiple(self):
        """Normal with multiple faces present"""
        bp = SharedPoint(self.get_point("bottom", 1))

        bp.add(self.bottom_face, 1)
        bp.add(self.get_face("right"), 1)

        np.testing.assert_array_almost_equal(bp.normal, f.unit_vector([1, 0, 1]))

    def test_is_single(self):
        sp = self.get_shared_point("bottom", 0)

        self.assertFalse(sp.is_shared)

    def test_is_shared(self):
        sp = self.get_shared_point("bottom", 0)
        sp.add(self.get_face("front"), 3)

        self.assertTrue(sp.is_shared)


class SharedpointStoreTests(ShellTestsBase):
    def setUp(self):
        super().setUp()

        self.sps = SharedPointStore()

    def test_sps_find_by_point_success(self):
        point = self.get_point("bottom", 0)
        shp = SharedPoint(point)
        shp.add(self.bottom_face, 0)

        self.sps.shared_points = [shp]

        self.assertEqual(self.sps.find_by_point(point), shp)

    def test_sps_find_by_point_failure(self):
        point = self.get_point("bottom", 0)
        shp = SharedPoint(point)
        shp.add(self.bottom_face, 0)

        self.sps.shared_points = [shp]

        with self.assertRaises(SharedPointNotFoundError):
            self.sps.find_by_point(self.get_point("bottom", 1))

    def test_sps_add_from_face_new(self):
        self.sps.add_from_face(self.get_face("bottom"), 0)

        self.assertEqual(len(self.sps.shared_points), 1)

    def test_sps_add_from_face_duplicate(self):
        """Return an existing object when adding the same bound point twice"""
        self.assertEqual(
            id(self.sps.add_from_face(self.get_face("bottom"), 0)),
            id(self.sps.add_from_face(self.get_face("bottom"), 0)),
        )


class AwareFaceTests(SharedPointTests):
    def get_aware_face(self, orient):
        face = self.get_face(orient)

        shared_points = []
        for i, point in enumerate(face.points):
            shared_point = SharedPoint(point)
            shared_point.add(face, i)
            shared_points.append(shared_point)

        return AwareFace(face, shared_points)

    def test_get_offset_points(self):
        # A simple test on a flat face; normal direction is
        # tested in BoundPoint
        face = self.get_face("bottom")
        aware_face = self.get_aware_face("bottom")

        face_points = [p.position for p in face.translate([0, 0, 1]).points]
        bound_offset = aware_face.get_offset_points(1)

        np.testing.assert_array_almost_equal(face_points, bound_offset)

    def test_get_offset_face(self):
        face_offset = self.get_aware_face("bottom").get_offset_face(1)
        bound_offset = self.get_face("bottom").translate([0, 0, 1])

        np.testing.assert_array_almost_equal(
            [p.position for p in face_offset.points], [p.position for p in bound_offset.points]
        )

    def test_is_solitary(self):
        aware_face = self.get_aware_face("bottom")

        self.assertTrue(aware_face.is_solitary)

    def test_is_not_solitary(self):
        aware_face = self.get_aware_face("bottom")
        aware_face.shared_points[0].add(self.get_face("front"), 3)

        self.assertFalse(aware_face.is_solitary)


class AwareFaceStoreTests(SharedPointTests):
    def get_aws(self, orients: List[OrientType]) -> AwareFaceStore:
        faces = [self.get_face(orient) for orient in orients]

        return AwareFaceStore(faces)

    def test_get_point_store_single(self):
        store = self.get_aws(["top"])

        self.assertEqual(len(store.point_store.shared_points), 4)

    def test_get_point_store_double_separate(self):
        store = self.get_aws(["top", "bottom"])

        self.assertEqual(len(store.point_store.shared_points), 8)

    def test_get_point_store_double_joined(self):
        store = self.get_aws(["top", "front"])

        self.assertEqual(len(store.point_store.shared_points), 6)

    def test_get_point_store_cube(self):
        store = self.get_aws(["front", "back", "left", "right", "top", "bottom"])

        self.assertEqual(len(store.point_store.shared_points), 8)

    @parameterized.expand(
        (
            (["bottom"],),
            (["bottom", "top"],),
            (["bottom", "top", "front"],),
            (["bottom", "top", "left", "right", "front", "back"],),
        )
    )
    def test_get_aware_faces(self, orients):
        store = self.get_aws(orients)

        self.assertEqual(len(store.aware_faces), len(orients))


class ShellTests(ShellTestsBase):
    def get_shell_faces(self, orients: List[OrientType]):
        box = Box([0, 0, 0], [1, 1, 1])

        faces = []

        for orient in orients:
            face = box.get_face(orient)
            if orient in ("front", "top", "left"):
                face.invert()

            faces.append(face)

        return faces

    def get_shell(self, orients: List[OrientType]):
        return Shell(self.get_shell_faces(orients), 0.5)

    @parameterized.expand(
        (
            (["top"],),
            (["bottom", "top"],),
            (["bottom", "top", "left", "right", "front", "back"],),
        )
    )
    def test_operations_count(self, orients):
        self.assertEqual(len(self.get_shell(orients).operations), len(orients))

    def test_chop(self):
        shell = self.get_shell(["left", "top"])
        shell.chop(count=10)

        self.assertEqual(len(shell.operations[0].chops[2]), 1)

    def test_set_outer_patch(self):
        orients = ["front", "right"]
        shell = self.get_shell(orients)
        shell.set_outer_patch("roof")

        for operation in shell.operations:
            self.assertEqual(operation.patch_names["top"], "roof")

    def test_chop_disconnected(self):
        shell = self.get_shell(["bottom", "top"])

        with self.assertRaises(DisconnectedChopError):
            shell.chop(coun=10)

    def test_grid(self):
        shell = self.get_shell(["front", "right", "left"])

        self.assertListEqual(shell.operations, shell.grid[0])
