from tests.fixtures.block import BlockTestCase

from classy_blocks.lists.block_list import BlockList

class BlockListTests(BlockTestCase):
    def setUp(self):
        self.bl = BlockList()

    def test_add(self):
        """Add a block and check the results"""
        self.bl.add(self.make_block(0))
        self.assertEqual(self.bl.blocks[0].index, 0)

        self.bl.add(self.make_block(1))
        self.assertEqual(self.bl.blocks[1].index, 1)


    # def test_merge_patches_duplicate(self):
    #     """duplicate coincident points on merged patches"""
    #     self.hexa_0.set_patch("right", "master")
    #     self.hexa_0.chop(1, count=10)

    #     self.hexa_1.set_patch("left", "slave")
    #     self.mesh.merge_patches("master", "slave")
    #     self.hexa_2.chop(0, count=10)

    #     self.prepare()

    #     # make sure hexa_0 and hexa_1 share no vertices
    #     set_0 = set(self.mesh.hexas[0].get_patch_sides("right"))
    #     set_1 = set(self.mesh.hexas[1].get_patch_sides("left"))

    #     self.assertTrue(set_0.isdisjoint(set_1))