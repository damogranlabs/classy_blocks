
from classy_blocks.define.block import Block
from classy_blocks.define.primitives import Vertex
from classy_blocks.util import functions as f
from classy_blocks.util import constants

from classy_blocks.process.lists.blocks import BlockList

from tests.fixtures import FixturedTestCase

class BlockListTests(FixturedTestCase):

    def test_add_block(self):
        """Add a single block"""
        self.prepare()

        self.assertEqual(len(self.mesh.blocks), 3)
        self.assertEqual(len(self.mesh.blocks.gradings), 3)
        self.assertEqual(len(self.mesh.blocks.neighbours), 3)

    def test_find_neighbour_success(self):
        """block_2 must copy block_1's cell count and grading on axis 0 and 2"""
        self.prepare()

        self.assertTrue(self.mesh.blocks.copy_grading(2, 0))
        self.assertTrue(self.mesh.blocks.copy_grading(2, 2))

    def test_find_neighbour_fail(self):
        """block_2 cannot copy cell count and grading from block_1 on axis 2"""
        self.block_1.chops = [[], [], []]

        self.assertRaises(Exception, self.prepare)

    def test_assign_neighbours(self):
        """assign neighbours to each block"""
        self.prepare()

        self.assertSetEqual(self.mesh.blocks.neighbours[0], {1, 2})
        self.assertSetEqual(self.mesh.blocks.neighbours[1], {0, 2})
        self.assertSetEqual(self.mesh.blocks.neighbours[2], {0, 1})

    def test_merge_patches_duplicate(self):
        """duplicate coincident points on merged patches"""
        self.block_0.set_patch("right", "master")
        self.block_0.chop(1, count=10)

        self.block_1.set_patch("left", "slave")
        self.mesh.merge_patches("master", "slave")
        self.block_2.chop(0, count=10)

        self.prepare()

        # make sure block_0 and block_1 share no vertices
        set_0 = set(self.mesh.blocks[0].get_patch_sides("right"))
        set_1 = set(self.mesh.blocks[1].get_patch_sides("left"))

        self.assertTrue(set_0.isdisjoint(set_1))