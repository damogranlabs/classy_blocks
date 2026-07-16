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

        mesh.assemble()
        mesh.grade()

        self.assertIn("simpleGrading ( 5 5 5 )", formats.format_block(mesh.blocks[0]))
        self.assertIn("simpleGrading ( 0.2 5 1 )", formats.format_block(mesh.blocks[1]))

    def test_autograde_with_merged_patch(self):
        # Bug case: face-merging a patch duplicates the slave block's corner
        # vertices. Those corners are also shared with non-merged neighbours,
        # so the duplication used to break grading coincidence and the count
        # could no longer be propagated to/from the neighbour.
        #
        #        +------+ boxC (on top of boxB, coincident, NOT merged)
        #        |      |
        # +------+------+
        # | boxA :| boxB |   boxA right <-> boxB left: face-merged
        # +------+------+
        box_a = cb.Box([0, 0, 0], [1, 1, 1])
        box_b = cb.Box([1, 0, 0], [2, 1, 1])
        box_c = cb.Box([1, 0, 1], [2, 1, 2])

        box_a.set_patch("right", "a_right")
        box_b.set_patch("left", "b_left")

        for axis in get_args(DirectionType):
            box_a.chop(axis, count=7)
            box_c.chop(axis, count=7)

        # leave box_b's axis 0 (x) unchopped: it must be inherited from box_c
        # through the shared top face whose corners were duplicated by the merge
        box_b.chop(1, count=7)
        box_b.chop(2, count=7)

        mesh = cb.Mesh()
        mesh.add(box_a)
        mesh.add(box_b)
        mesh.add(box_c)
        mesh.merge_patches("a_right", "b_left")

        mesh.assemble()
        mesh.grade()

        # box_b's x-count was inherited from the coincident (non-merged) box_c
        self.assertEqual(mesh.blocks[1].axes[0].count, 7)
