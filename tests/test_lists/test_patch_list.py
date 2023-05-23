from tests.fixtures.block import BlockTestCase

from classy_blocks.lists.patch_list import PatchList


class PatchListTests(BlockTestCase):
    def setUp(self):
        self.plist = PatchList()
        self.vertices = self.make_vertices(0)
        self.loft = self.make_loft(0)
        self.name = 'vessel'
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

    def test_is_slave(self):
        """Find a patch among slave patches"""
        self.plist.merge("master", "slave1")
        self.plist.merge("master", "slave2")

        self.assertTrue(self.plist.is_slave("slave1"))
        self.assertTrue(self.plist.is_slave("slave2"))
        self.assertFalse(self.plist.is_slave("slave3"))

    def test_add(self):
        """Add an Operation"""
        self.plist.add(self.vertices, self.loft)

        self.assertEqual(len(self.plist.patches[self.name].sides), 5)

    def test_set_default(self):
        """Set default patch"""
        self.plist.set_default('walls', 'wall')
        description = self.plist.description

        self.assertIn('defaultPatch', description)
        self.assertIn('type wall;', description)
        self.assertIn('name walls;', description)
    
    def test_modify_settings(self):
        """Modify patch settings"""
        self.plist.add(self.vertices, self.loft)
        self.plist.modify(self.name, 'cyclic', ['neighbourPatch anti_vessel'])

        description = self.plist.description

        self.assertIn('type cyclic;', description)
        self.assertIn('neighbourPatch anti_vessel;', description)


    def test_master_patches(self):
        self.plist.merge('master1', 'slave1')
        self.plist.merge('master2', 'slave2')

        self.assertSetEqual(self.plist.master_patches, {'master1', 'master2'})