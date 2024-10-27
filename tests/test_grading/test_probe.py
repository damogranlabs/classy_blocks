from typing import get_args

from parameterized import parameterized

from classy_blocks.grading.autograding.probe import Probe, get_block_from_axis
from classy_blocks.mesh import Mesh
from classy_blocks.types import DirectionType
from tests.test_grading.test_autograde import AutogradeTestsBase


class ProbeTests(AutogradeTestsBase):
    def test_block_from_axis_fail(self):
        mesh_1 = self.mesh
        mesh_1.add(self.get_stack())
        mesh_1.assemble()

        mesh_2 = Mesh()
        mesh_2.add(self.get_stack())
        mesh_2.assemble()

        with self.assertRaises(RuntimeError):
            get_block_from_axis(mesh_1, mesh_2.blocks[0].axes[0])

    @parameterized.expand((("min", 0.196889), ("max", 0.8), ("avg", 0.470923)))
    def test_get_row_length(self, take, length):
        self.mesh.add(self.get_frustum())
        self.mesh.assemble()

        probe = Probe(self.mesh)
        row = probe.get_rows(0)[0]

        self.assertAlmostEqual(row.get_length(take), length, places=4)

    @parameterized.expand(((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 2)))
    def test_get_blocks_on_layer(self, block, axis):
        self.mesh.add(self.get_stack())
        self.mesh.assemble()

        probe = Probe(self.mesh)
        blocks = probe.get_row_blocks(self.mesh.blocks[block], axis)

        self.assertEqual(len(blocks), 9)

    @parameterized.expand(
        (
            (0,),
            (1,),
            (2,),
            (3,),
            (4,),
            (5,),
            (6,),
            (7,),
            (8,),
            (9,),
            (10,),
            (11,),
        )
    )
    def test_block_from_axis(self, index):
        self.mesh.add(self.get_cylinder())
        self.mesh.assemble()

        for axis in get_args(DirectionType):
            block = self.mesh.blocks[index]

            self.assertEqual(block, get_block_from_axis(self.mesh, block.axes[axis]))

    @parameterized.expand(((0,), (1,), (2,)))
    def test_get_layers(self, axis):
        self.mesh.add(self.get_stack())
        self.mesh.assemble()

        probe = Probe(self.mesh)
        layers = probe.get_rows(axis)

        self.assertEqual(len(layers), 3)

    @parameterized.expand(
        (
            # axis, layer, block indexes
            (0, 0, {5, 0, 3, 10}),
            (0, 1, {6, 1, 2, 9}),
            (0, 2, {4, 5, 6, 7, 8, 9, 10, 11}),
            (1, 0, {7, 1, 0, 4}),
            (1, 1, {8, 2, 3, 11}),
            (2, 0, set(range(12))),
        )
    )
    def test_get_blocks_cylinder(self, axis, row, blocks):
        self.mesh.add(self.get_cylinder())
        self.mesh.assemble()

        probe = Probe(self.mesh)
        indexes = set()

        for block in probe.catalogue.rows[axis][row].blocks:
            indexes.add(block.index)

        self.assertSetEqual(indexes, blocks)
