from classy_blocks.items.patch import Patch
from classy_blocks.items.side import Side
from classy_blocks.types import OrientType
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

    def test_description(self):
        """String output format"""
        patch = self.patch
        patch.add_side(self.get_side(0, "bottom"))
        patch.add_side(self.get_side(0, "left"))
        patch.kind = "wall"

        expected = (
            "\ttest\n"
            + "\t{\n"
            + "\t\ttype wall;\n"
            + "\t\tfaces\n"
            + "\t\t(\n"
            + "\t\t\t(0 1 2 3)\n"
            + "\t\t\t(4 0 3 7)\n"
            + "\t\t);\n"
            + "\t}\n"
        )

        self.assertEqual(patch.description, expected)

    def test_options(self):
        """Add an option and check it's in description"""
        option = "neighbourPatch left"

        patch = self.patch
        patch.settings.append(option)

        self.assertTrue(f"\t{option};" in patch.description)
