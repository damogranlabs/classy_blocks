import collections
import unittest

import numpy as np

from classy_blocks.construct.flat.sketches.grid import Grid
from classy_blocks.construct.operations.box import Box
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.construct.stack import ExtrudedStack
from classy_blocks.grading.graders.fixed import FixedCountGrader
from classy_blocks.grading.graders.inflation import InflationGrader
from classy_blocks.grading.graders.simple import SimpleGrader
from classy_blocks.mesh import Mesh
from classy_blocks.write import formats


class RespectManualChopsTests(unittest.TestCase):
    """Auto graders must respect user's manually-created chops (which effectively
    set the count of a row) and only add chops on rows that are still missing."""

    def test_single_block(self):
        # a manual chop on one direction, the other two filled by the auto grader
        box = Box([0, 0, 0], [1, 1, 1])
        box.chop(0, count=13)

        mesh = Mesh()
        mesh.add(box)

        FixedCountGrader(mesh, 5).grade()

        block = mesh.blocks[0]
        self.assertEqual(block.axes[0].count, 13)  # manual
        self.assertEqual(block.axes[1].count, 5)  # auto
        self.assertEqual(block.axes[2].count, 5)  # auto

    def test_graded_chop_preserved(self):
        # a non-uniform (graded) manual chop must keep its grading, not be
        # flattened by the auto grader
        box = Box([0, 0, 0], [1, 1, 1])
        box.chop(0, start_size=0.01, end_size=0.2)

        mesh = Mesh()
        mesh.add(box)

        SimpleGrader(mesh, 0.1).grade()

        block = mesh.blocks[0]
        self.assertIn("simpleGrading ( 20.0 1 1 )", formats.format_block(block))

    def test_multi_block_row(self):
        # two blocks stacked in z share a single row in x (via coincidence);
        # chopping the bottom block's x must propagate to the top one, not be
        # overwritten by the auto grader
        bottom = Box([0, 0, 0], [1, 1, 1])
        top = Box([0, 0, 1], [1, 1, 2])
        bottom.chop(0, count=13)

        mesh = Mesh()
        mesh.add(bottom)
        mesh.add(top)

        FixedCountGrader(mesh, 5).grade()

        self.assertEqual(mesh.blocks[0].axes[0].count, 13)
        self.assertEqual(mesh.blocks[1].axes[0].count, 13)  # inherited, not auto 5
        self.assertEqual(mesh.blocks[0].axes[1].count, 5)
        self.assertEqual(mesh.blocks[0].axes[2].count, 5)

    def test_extruded_grid(self):
        # a manually chopped row in a 4x4 grid stack stays put; the rest is auto
        base = Grid([0, 0, 0], [1, 1, 0], 4, 4)
        stack = ExtrudedStack(base, 1, 4)
        stack.operations[0].chop(0, count=17)

        mesh = Mesh()
        mesh.add(stack)

        FixedCountGrader(mesh, 5).grade()

        self.assertEqual(sorted({b.axes[0].count for b in mesh.blocks}), [5, 17])
        self.assertEqual(sorted({b.axes[1].count for b in mesh.blocks}), [5])
        self.assertEqual(sorted({b.axes[2].count for b in mesh.blocks}), [5])

    def test_flipped_blocks(self):
        # same as above but with some operations flipped upside-down, to exercise
        # row alignment
        base = Grid([0, 0, 0], [1, 1, 0], 4, 4)
        stack = ExtrudedStack(base, 1, 4)
        for i in (5, 6, 8, 9):
            stack.operations[i].rotate(np.pi, [0, 1, 0])
        stack.operations[5].chop(0, count=17)

        mesh = Mesh()
        mesh.add(stack)

        FixedCountGrader(mesh, 5).grade()

        self.assertEqual(sorted({b.axes[0].count for b in mesh.blocks}), [5, 17])
        self.assertEqual(sorted({b.axes[1].count for b in mesh.blocks}), [5])
        self.assertEqual(sorted({b.axes[2].count for b in mesh.blocks}), [5])

    def test_inflation_grader_wall_normal(self):
        # the inflation grader must not override a manually chopped (uniform)
        # wall-normal direction with an inflation layer
        box = Box([0, 0, 0], [1, 1, 1])
        for orient in ("left", "right", "front", "back", "top", "bottom"):
            box.set_patch(orient, "walls")
        box.chop(1, count=8)

        mesh = Mesh()
        mesh.add(box)
        mesh.modify_patch("walls", "wall")

        InflationGrader(mesh, 1e-3, 0.1).grade()

        block = mesh.blocks[0]
        # a single uniform chop in the manually-graded direction
        self.assertEqual(block.axes[1].count, 8)
        self.assertEqual(len(block.axes[1].wires.wires[0].grading.chops), 1)
        # the un-chopped directions got a full inflation stack
        self.assertGreater(len(block.axes[0].wires.wires[0].grading.chops), 1)

    def test_cylinder_shape(self):
        # a shape spreads a single chop over many operations; auto grader fills
        # only the un-chopped (axial) direction
        cylinder = Cylinder([0, 0, 0], [0, 0, 2], [1, 0, 0])
        cylinder.chop_radial(count=6)
        cylinder.chop_tangential(count=8)

        mesh = Mesh()
        mesh.add(cylinder)

        SimpleGrader(mesh, 0.2).grade()

        by_axis = collections.defaultdict(set)
        for block in mesh.blocks:
            for axis in range(3):
                by_axis[axis].add(block.axes[axis].count)

        # radial (6) and tangential (8) counts come from the user; both appear on
        # axis 0 because core/shell blocks map that axis differently
        self.assertEqual(sorted(by_axis[0]), [6, 8])
        self.assertEqual(sorted(by_axis[1]), [8])


class RespectManualChopsMergedTests(unittest.TestCase):
    """Auto grading on a mesh that uses face-merging. The merged interface keeps
    the two sides independent, while non-merged coincident neighbours still share
    counts (see also tests/test_bugs/test_grading.py)."""

    def get_merged_boxes(self) -> tuple[Box, Box, Box]:
        #        +------+ box_c (on top of box_b, coincident, NOT merged)
        #        |      |
        # +------+------+
        # | box_a |box_b|   box_a right <-> box_b left: face-merged
        # +------+------+
        box_a = Box([0, 0, 0], [1, 1, 1])
        box_b = Box([1, 0, 0], [2, 1, 1])
        box_c = Box([1, 0, 1], [2, 1, 2])

        box_a.set_patch("right", "a_right")
        box_b.set_patch("left", "b_left")

        return box_a, box_b, box_c

    def get_mesh(self, box_a: Box, box_b: Box, box_c: Box) -> Mesh:
        mesh = Mesh()
        mesh.add(box_a)
        mesh.add(box_b)
        mesh.add(box_c)
        mesh.merge_patches("a_right", "b_left")

        return mesh

    def test_inherited_grading_not_inverted(self):
        # box_c is manually chopped in x and box_b inherits that grading through
        # the shared (non-merged) top face, whose corners the merge duplicated;
        # the duplication must not make the two axes look oppositely oriented
        box_a, box_b, box_c = self.get_merged_boxes()
        box_c.chop(0, start_size=0.02, end_size=0.4)

        mesh = self.get_mesh(box_a, box_b, box_c)
        FixedCountGrader(mesh, 5).grade()

        self.assertIn("simpleGrading ( 20.0 1 1 )", formats.format_block(mesh.blocks[2]))  # box_c, manual
        self.assertIn("simpleGrading ( 20.0 1 1 )", formats.format_block(mesh.blocks[1]))  # box_b, not 0.05

    def test_merged_interface_stays_independent(self):
        # a manual chop on the slave's in-merge-face direction must NOT leak to
        # the master side across the merged interface. Use only the two merged
        # boxes so the sole connection between them is the merged face itself.
        box_a = Box([0, 0, 0], [1, 1, 1])
        box_b = Box([1, 0, 0], [2, 1, 1])
        box_a.set_patch("right", "a_right")
        box_b.set_patch("left", "b_left")
        box_b.chop(1, count=12)  # in-face direction

        mesh = Mesh()
        mesh.add(box_a)
        mesh.add(box_b)
        mesh.merge_patches("a_right", "b_left")

        FixedCountGrader(mesh, 5).grade()

        self.assertEqual(mesh.blocks[1].axes[1].count, 12)  # box_b, manual
        self.assertEqual(mesh.blocks[0].axes[1].count, 5)  # box_a, independent


class PartialOverlapTests(unittest.TestCase):
    def test_two_box(self):
        box_a = Box([0, 0, 0], [1, 1, 1])
        box_b = Box([1, 0, 0], [2, 2, 1])
        box_a.set_patch("right", "a_right")
        box_b.set_patch("left", "b_left")
        box_b.chop(1, count=12)  # in-face direction

        mesh = Mesh()
        mesh.add(box_a)
        mesh.add(box_b)
        mesh.merge_patches("a_right", "b_left")

        FixedCountGrader(mesh, 5).grade()

        self.assertEqual(mesh.blocks[1].axes[1].count, 12)  # box_b, manual
        self.assertEqual(mesh.blocks[0].axes[1].count, 5)  # box_a, independent


if __name__ == "__main__":
    unittest.main()
