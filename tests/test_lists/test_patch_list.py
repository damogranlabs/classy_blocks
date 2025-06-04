from classy_blocks.lists.patch_list import PatchList
from tests.fixtures.block import BlockTestCase


class PatchListTests(BlockTestCase):
    def setUp(self):
        self.plist = PatchList()
        self.vertices = self.make_vertices(0)
        self.loft = self.make_loft(0)
        self.name = "vessel"
        self.loft.set_patch(["left", "bottom", "front", "back", "right"], self.name)

    def test_get_new(self):
        """Add a new entry to patches dict when fetching
        a patch not entered before"""
        self.assertFalse("test" in self.plist.patches)
        self.plist.get("test")
        self.assertTrue("test" in self.plist.patches)

    def test_get_existing(self):
        """Fetch the same patch twice"""
        self.plist.get("test")

        self.assertEqual(self.plist.get("test"), self.plist.get("test"))

    def test_modify_type(self):
        """Modify patch type"""
        self.plist.modify("walls", "wall")
        self.assertEqual(self.plist.get("walls").kind, "wall")

    def test_add(self):
        """Add an Operation"""
        self.plist.add(self.vertices, self.loft)

        self.assertEqual(len(self.plist.patches[self.name].sides), 5)
