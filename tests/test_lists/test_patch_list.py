from tests.fixtures.block import BlockTestCase

from classy_blocks.lists.patch_list import PatchList


class PatchListTests(BlockTestCase):
    def setUp(self):
        self.plist = PatchList()

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

    def test_modify(self):
        """Modify patch type"""
        self.plist.modify("walls", "wall")
        self.assertEqual(self.plist.get("walls").kind, "wall")

    def test_is_slave(self):
        """Find a patch among slave patches"""
        self.plist.merge("master", "slave1")
        self.plist.merge("master", "slave2")

        self.assertTrue(self.plist.is_slave("slave1"))
        self.assertTrue(self.plist.is_slave("slave2"))
        self.assertFalse(self.plist.is_slave("slave3"))

    def test_add(self):
        """Add an Operation"""
        vertices = self.make_vertices(0)
        loft = self.make_loft(0)
        loft.set_patch(["left", "bottom", "front", "back", "right"], "vessel")

        self.plist.add(vertices, loft)

        self.assertEqual(len(self.plist.patches["vessel"].sides), 5)
