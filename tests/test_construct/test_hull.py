from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shapes.hull import BoundPoint, FusedFaceCollection
from tests.fixtures.block import DataTestCase


class HullTests(DataTestCase):
    def setUp(self):
        data = self.get_single_data(0)

        self.bottom_face = Face(data.points[:4])
        self.top_face = Face(data.points[4:])

        self.loft = Loft(self.bottom_face, self.top_face)

    def get_face(self, orient) -> Face:
        return self.loft.get_face(orient)

    def test_bound_point_equal(self):
        face1 = self.bottom_face
        bp1 = BoundPoint(face1.points[1])
        bp1.add(face1, 1)

        face2 = self.loft.get_face("right")
        bp2 = BoundPoint(face2.points[1])
        bp2.add(face2, 1)

        self.assertEqual(bp1, bp2)

    def test_ffcollection_single(self):
        """A fused face collection with a single face"""
        ffc = FusedFaceCollection([self.bottom_face])

        self.assertEqual(len(ffc.bound_points), 4)

    def test_ffcollection_disjoint(self):
        """Add two non-touching faces"""
        ffc = FusedFaceCollection([self.bottom_face, self.top_face])

        self.assertEqual(len(ffc.bound_points), 8)

    def test_ffcollection_fused(self):
        """Add two touching faces to collection"""
        ffc = FusedFaceCollection([self.bottom_face, self.loft.get_face("right")])

        self.assertEqual(len(ffc.bound_points), 6)

    def test_bound_point_indexes_single(self):
        """A solitary bound point at a corner of a single face"""
        ffc = FusedFaceCollection([self.bottom_face])

        self.assertListEqual(ffc.bound_points[0].indexes, [0])

    def test_bound_point_indexes_double(self):
        """Bound point at joint of two faces"""
        ffc = FusedFaceCollection([self.bottom_face, self.loft.get_face("right")])

        self.assertListEqual(ffc.bound_points[1].indexes, [1, 1])

    def test_bount_point_indexes_triple(self):
        """Bound point at joint of three faces"""
        ffc = FusedFaceCollection([self.bottom_face, self.loft.get_face("right"), self.loft.get_face("front")])

        self.assertListEqual(ffc.bound_points[1].indexes, [1, 1, 1])
