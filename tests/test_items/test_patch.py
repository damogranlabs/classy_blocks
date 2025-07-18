from classy_blocks.cbtyping import OrientType
from classy_blocks.items.patch import Patch
from classy_blocks.items.side import Side
from tests.fixtures.block import BlockTestCase


class PatchTests(BlockTestCase):
    def setUp(self):
        self.name = "test"

    @property
    def patch(self) -> Patch:
        """The test subject"""
        return Patch(self.name)

    def get_side(self, index: int, orient: OrientType) -> Side:
        """Creates a side on 'orient' from block at 'index'"""
        return Side(orient, self.make_vertices(index))

    def test_add_side(self):
        """Add a single side to block"""
        patch = self.patch
        patch.add_side(self.get_side(0, "left"))

        self.assertEqual(len(patch.sides), 1)

    def test_add_equal_sides(self):
        """Add the same side twice"""
        patch = self.patch
        patch.add_side(self.get_side(0, "left"))

        with self.assertWarns(Warning):
            patch.add_side(self.get_side(0, "left"))

        self.assertEqual(len(patch.sides), 1)
