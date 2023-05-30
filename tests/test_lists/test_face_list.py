from parameterized import parameterized

from classy_blocks.items.side import Side
from classy_blocks.lists.face_list import FaceList, ProjectedFace
from classy_blocks.types import OrientType
from tests.fixtures.block import BlockTestCase


class ProjectedFaceTests(BlockTestCase):
    @property
    def side_1(self) -> Side:
        return Side("right", self.make_vertices(0))

    @property
    def side_2(self) -> Side:
        return Side("left", self.make_vertices(1))

    def test_equal(self):
        pface_1 = ProjectedFace(self.side_1, "terrain")
        pface_2 = ProjectedFace(self.side_2, "geometry")

        self.assertEqual(pface_1, pface_2)

    def test_not_equal(self):
        pface_1 = ProjectedFace(self.side_1, "terrain")

        # change a vertex to a third-party
        side_2 = self.side_2
        side_2.vertices[1] = self.make_vertices(2)[-1]

        pface_2 = ProjectedFace(side_2, "terrain")

        self.assertNotEqual(pface_1, pface_2)

    def test_description(self):
        pface = ProjectedFace(self.side_1, "terrain")

        self.assertEqual(pface.description, "\tproject (5 1 2 6) terrain\n")


class FaceListTests(BlockTestCase):
    def setUp(self):
        self.flist = FaceList()

        self.index = 0
        self.vertices = self.make_vertices(self.index)
        self.loft = self.make_loft(self.index)

    def get_side(self, block: int, orient: OrientType) -> Side:
        return Side(orient, self.make_vertices(block))

    def test_find_existing_success(self):
        self.loft.project_side("left", "geometry")
        self.flist.add(self.vertices, self.loft)

        side = self.get_side(0, "left")
        self.assertTrue(self.flist.find_existing(side))

    def test_find_existing_fail(self):
        self.flist.add(self.vertices, self.loft)

        side = self.get_side(0, "left")
        self.assertFalse(self.flist.find_existing(side))

    @parameterized.expand(
        (
            ("left",),
            ("right",),
            ("front",),
            ("back",),
            ("top",),
            ("bottom",),
        )
    )
    def test_capture_sides(self, orient):
        """Capture loft's projected side faces"""
        self.loft.project_side(orient, "terrain")

        self.flist.add(self.vertices, self.loft)

        self.assertEqual(self.flist.faces[0].geometry, "terrain")

    def test_description(self):
        self.loft.project_side("bottom", "terrain")
        self.flist.add(self.vertices, self.loft)

        self.assertEqual(self.flist.description, "faces\n(\n\tproject (0 1 2 3) terrain\n);\n\n")
