import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shapes.shell import (
    BoundFace,
    BoundPoint,
    BoundPointCollection,
    BoundPointNotFoundError,
    PointNotCoincidentError,
)
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


class BoundPointTests(ShellTestsBase):
    def test_bound_point_equal(self):
        face1 = self.bottom_face
        bp1 = BoundPoint(face1.points[1])
        bp1.add(face1, 1)

        face2 = self.loft.get_face("right")
        bp2 = BoundPoint(face2.points[1])
        bp2.add(face2, 1)

        self.assertEqual(bp1, bp2)

    def test_bound_point_add(self):
        """Add a legit face/index to a BoundPoint"""
        point = self.get_point("bottom", 0)
        bp = BoundPoint(point)

        bp.add(self.bottom_face, 0)

        self.assertEqual(len(bp.faces), 1)

    def test_bound_point_duplicated(self):
        """Do not add duplicate faces"""
        face = self.get_face("bottom")
        point = face.points[0]

        bp = BoundPoint(point)

        bp.add(face, 0)
        bp.add(face, 0)

        self.assertEqual(len(bp.faces), 1)

    def test_bound_point_noncoincident(self):
        """Raise an exception when trying to add a bound point at a different location"""
        bp = BoundPoint(self.get_point("bottom", 0))

        with self.assertRaises(PointNotCoincidentError):
            bp.add(self.bottom_face, 1)

    def test_bound_point_normal_single(self):
        """Normal when a single face is present in a bound point"""
        point = self.get_point("bottom", 0)
        bp = BoundPoint(point)
        bp.add(self.bottom_face, 0)

        np.testing.assert_array_almost_equal(bp.normal, self.bottom_face.normal)

    def test_bound_point_normal_multiple(self):
        """Normal with multiple faces present"""
        bp = BoundPoint(self.get_point("bottom", 1))

        bp.add(self.bottom_face, 1)
        bp.add(self.get_face("right"), 1)

        np.testing.assert_array_almost_equal(bp.normal, f.unit_vector([1, 0, 1]))


class BoundPointCollectionTests(ShellTestsBase):
    def setUp(self):
        super().setUp()

        self.bpc = BoundPointCollection()

    def test_bpc_find_by_point_success(self):
        point = self.get_point("bottom", 0)
        bp = BoundPoint(point)
        bp.add(self.bottom_face, 0)

        self.bpc.bound_points = [bp]

        self.assertEqual(self.bpc.find_by_point(point), bp)

    def test_bpc_find_by_point_failure(self):
        point = self.get_point("bottom", 0)
        bp = BoundPoint(point)
        bp.add(self.bottom_face, 0)

        self.bpc.bound_points = [bp]

        with self.assertRaises(BoundPointNotFoundError):
            self.bpc.find_by_point(self.get_point("bottom", 1))

    def test_bpc_add_from_face_new(self):
        self.bpc.add_from_face(self.get_face("bottom"), 0)

        self.assertEqual(len(self.bpc.bound_points), 1)

    def test_bpc_add_from_face_duplicate(self):
        """Return an existing object when adding the same bound point twice"""
        self.assertEqual(
            id(self.bpc.add_from_face(self.get_face("bottom"), 0)),
            id(self.bpc.add_from_face(self.get_face("bottom"), 0)),
        )


class BoundFaceTests(BoundPointTests):
    def get_bound_face(self, orient):
        face = self.get_face(orient)

        bound_points = []
        for i, point in enumerate(face.points):
            bound_point = BoundPoint(point)
            bound_point.add(face, i)
            bound_points.append(bound_point)

        return BoundFace(face, bound_points)

    def test_get_offset_points(self):
        # A simple test on a flat face; normal direction is
        # tested in BoundPoint
        face = self.get_face("bottom")
        bound_face = self.get_bound_face("bottom")

        face_points = [p.position for p in face.translate([0, 0, 1]).points]
        bound_offset = bound_face.get_offset_points(1)

        np.testing.assert_array_almost_equal(face_points, bound_offset)

    def test_get_offset_face(self):
        face_offset = self.get_bound_face("bottom").get_offset_face(1)
        bound_offset = self.get_face("bottom").translate([0, 0, 1])

        np.testing.assert_array_almost_equal(
            [p.position for p in face_offset.points], [p.position for p in bound_offset.points]
        )


class BoundFaceCollectionTests(BoundPointTests):
    pass


class ShellTests:
    # TODO: common points tests
    pass
