import unittest

import numpy as np

from classy_blocks.construct.operations.box import Box
from classy_blocks.grading.graders.inflation import InflationGrader
from classy_blocks.mesh import Mesh


class InflationGraderTests(unittest.TestCase):
    def get_grader(self, mesh: Mesh) -> InflationGrader:
        return InflationGrader(mesh, 0.002, 0.1, 1.2, 30)

    def test_flipped_blocks(self):
        # create two boxes side by side, one is flipped
        # upside-down;
        mesh = Mesh()

        box_1 = Box([0, 0, 0], [1, 1, 1])
        box_1.set_patch("bottom", "wall")

        box_2 = Box([1, 0, 0], [2, 1, 1])
        box_2.rotate(np.pi, [0, 1, 0])
        box_2.set_patch("top", "wall")

        mesh.add(box_1)
        mesh.add(box_2)

        # the bottom-most patch (as-is) is wall
        mesh.modify_patch("wall", "wall")

        grader = self.get_grader(mesh)
        grader.grade()

        # now make sure the blocks are not double graded
        self.assertEqual(len(mesh.blocks[0].axes[2].wires.wires[0].grading.chops), 3)
        self.assertEqual(len(mesh.blocks[1].axes[2].wires.wires[0].grading.chops), 3)
