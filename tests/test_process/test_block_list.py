from tests.fixtures import FixturedTestCase

from classy_blocks.lists.block_list import BlockList

class BlockListTests(FixturedTestCase):
    def setUp(self):
        self.blocks = self.get_blocks()
        self.bl = BlockList()

    def test_add(self):
        """Add a block and check the results"""
        self.bl.add(self.blocks[0])
        self.assertEqual(self.bl.data[0].index, 0)

        self.bl.add(self.blocks[1])
        self.assertEqual(self.bl.data[1].index, 1)


    # def test_find_neighbour_success(self):
    #     """hexa_2 must copy hexa_1's cell count and grading on axis 0 and 2"""
    #     self.prepare()

    #     self.assertTrue(self.mesh.hexas.copy_grading(2, 0))
    #     self.assertTrue(self.mesh.hexas.copy_grading(2, 2))

    # def test_find_neighbour_fail(self):
    #     """hexa_2 cannot copy cell count and grading from hexa_1 on axis 2"""
    #     self.hexa_1.chops = [[], [], []]

    #     self.assertRaises(Exception, self.prepare)

    # def test_assign_neighbours(self):
    #     """assign neighbours to each hexa"""
    #     self.prepare()

    #     self.assertSetEqual(self.mesh.hexas.neighbours[0], {1, 2})
    #     self.assertSetEqual(self.mesh.hexas.neighbours[1], {0, 2})
    #     self.assertSetEqual(self.mesh.hexas.neighbours[2], {0, 1})

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