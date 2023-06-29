import numpy as np

from classy_blocks.mesh import Mesh
from tests.fixtures.block import BlockTestCase


class PropagationTests(BlockTestCase):
    def setUp(self):
        self.mesh = Mesh()

    def test_propagate_normal(self):
        """Propagate grading from a block to another"""
        op_0 = self.make_loft(0)
        op_0.chop(0, count=10)
        op_0.chop(1, count=20, total_expansion=5)
        op_0.chop(2, count=10)
        self.mesh.add(op_0)

        op_1 = self.make_loft(1)
        op_1.chop(0, count=10)

        self.mesh.add(op_1)

        self.mesh.assemble()
        self.mesh.block_list.propagate_gradings()

        self.assertListEqual(
            self.mesh.block_list.blocks[0].axes[1].grading.specification,
            self.mesh.block_list.blocks[1].axes[1].grading.specification,
        )

    def test_propagate_upsidedown(self):
        """Propagate grading from a block to an overturned block; invert grading"""
        op_0 = self.make_loft(0)
        op_0.chop(0, count=10)
        op_0.chop(1, count=20, total_expansion=5)
        op_0.chop(2, count=10)
        self.mesh.add(op_0)

        op_1 = self.make_loft(1).rotate(np.pi, [1, 0, 0])
        op_1.chop(0, count=10)

        self.mesh.add(op_1)

        self.mesh.assemble()
        self.mesh.block_list.propagate_gradings()

        self.assertListEqual(
            self.mesh.block_list.blocks[0].axes[1].grading.specification,
            self.mesh.block_list.blocks[1].axes[1].grading.inverted.specification,
        )
