import unittest
from typing import get_args

from parameterized import parameterized

from classy_blocks.construct.flat.sketches.grid import Grid
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.construct.stack import ExtrudedStack
from classy_blocks.grading.autograding.probe import Probe, get_block_from_axis
from classy_blocks.mesh import Mesh
from classy_blocks.types import DirectionType


class AutogradeTestsBase(unittest.TestCase):
    def get_stack(self) -> ExtrudedStack:
        # create a simple 3x3 grid for easy navigation
        base = Grid([0, 0, 0], [1, 1, 0], 3, 3)
        return ExtrudedStack(base, 1, 3)

    def get_cylinder(self) -> Cylinder:
        return Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])

    def setUp(self):
        self.mesh = Mesh()


class ProbeTests(AutogradeTestsBase):
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
