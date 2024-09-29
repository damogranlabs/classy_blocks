import unittest

from parameterized import parameterized

from classy_blocks.construct.flat.sketches.grid import Grid
from classy_blocks.construct.stack import ExtrudedStack
from classy_blocks.grading.autograding.probe import Probe
from classy_blocks.mesh import Mesh


class AutogradeTestsBase(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh()

        # create a simple 3x3 grid for easy navigation
        base = Grid([0, 0, 0], [1, 1, 0], 3, 3)
        self.stack = ExtrudedStack(base, 1, 3)

        self.mesh.add(self.stack)
        self.mesh.assemble()


class ProbeTests(AutogradeTestsBase):
    @parameterized.expand(((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 2)))
    def test_get_blocks_on_layer(self, block, axis):
        probe = Probe(self.mesh)

        blocks = probe._get_blocks_on_layer(self.mesh.blocks[block], axis)
        self.assertEqual(len(blocks), 9)

    @parameterized.expand(((0,), (1,), (2,)))
    def test_get_layers(self, axis):
        probe = Probe(self.mesh)

        layers = probe.get_layers(axis)

        self.assertEqual(len(layers), 3)
