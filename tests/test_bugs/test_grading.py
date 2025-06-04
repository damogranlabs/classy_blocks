import unittest
from typing import get_args

import numpy as np

import classy_blocks as cb
from classy_blocks.cbtyping import DirectionType
from classy_blocks.write import formats


class GradingBugTests(unittest.TestCase):
    def test_invert_grading(self):
        # Bug case; two blocks with separate bottom faces share the same top face.
        # Grading in one direction must be inverted
        #          /|\
        #        / / \ \
        #      /  /   \  \
        #    /   /     \   \
        #  /____/   ^   \____\
        #  base  common  neighbour

        box = cb.Box([0, 0, 0], [1, 1, 1])

        base_face = box.bottom_face
        neighbour_face = base_face.copy().translate([4, 0, 0])
        common_face = base_face.copy().rotate(np.pi / 2, [0, 1, 0]).translate([2, 0, 2])

        left_loft = cb.Loft(base_face, common_face)
        right_loft = cb.Loft(neighbour_face, common_face.copy().shift(2).invert())

        for axis in get_args(DirectionType):
            left_loft.chop(axis, start_size=0.05, total_expansion=5)

        right_loft.chop(2, count=10)

        mesh = cb.Mesh()
        mesh.add(left_loft)
        mesh.add(right_loft)

        dump = mesh.assemble()
        dump.block_list.assemble()

        self.assertIn("simpleGrading ( 5 5 5 )", formats.format_block(mesh.blocks[0]))
        self.assertIn("simpleGrading ( 0.2 5 1 )", formats.format_block(mesh.blocks[1]))
