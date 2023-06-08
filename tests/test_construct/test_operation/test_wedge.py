import unittest

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.wedge import Wedge


class WedgeTests(unittest.TestCase):
    def setUp(self):
        self.points = [[0, 1, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0]]

    @property
    def base(self) -> Face:
        return Face(self.points)

    @property
    def wedge(self) -> Wedge:
        return Wedge(self.base)

    def test_wedge_construction_mirror(self):
        """Check that bottom and top faces are mirrored around XY-plane"""
        wedge = self.wedge

        for i in range(4):
            self.assertAlmostEqual(wedge.bottom_face.points[i].position[2], -wedge.top_face.points[i].position[2])

    def test_wedge_construction_revolve(self):
        """Check that y-values of face points are the same"""
        wedge = self.wedge

        for i in range(4):
            self.assertAlmostEqual(wedge.bottom_face.points[i].position[1], wedge.top_face.points[i].position[1])

    def test_inner_patch(self):
        wedge = self.wedge
        wedge.set_inner_patch("test")

        self.assertEqual(wedge.patch_names["front"], "test")

    def test_outer_patch(self):
        wedge = self.wedge
        wedge.set_outer_patch("test")

        self.assertEqual(wedge.patch_names["back"], "test")

    def test_chop(self):
        """Automatically chop to count=1 in revolved direction"""
        self.assertEqual(self.wedge.chops[2][0].count, 1)
