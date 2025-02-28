import unittest

import numpy as np

import classy_blocks as cb
from classy_blocks.construct.flat.sketches.grid import Grid
from classy_blocks.construct.operations.box import Box
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.construct.shapes.frustum import Frustum
from classy_blocks.construct.stack import ExtrudedStack
from classy_blocks.grading.autograding.fixed.grader import FixedCountGrader
from classy_blocks.grading.autograding.simple.grader import SimpleGrader
from classy_blocks.grading.autograding.smooth.grader import SmoothGrader
from classy_blocks.mesh import Mesh


class AutogradeTestsBase(unittest.TestCase):
    def get_stack(self) -> ExtrudedStack:
        # create a simple 4x4 grid for easy navigation
        base = Grid([0, 0, 0], [1, 1, 0], 4, 4)
        return ExtrudedStack(base, 1, 4)

    def get_flipped_stack(self) -> ExtrudedStack:
        stack = self.get_stack()

        for i in (5, 6, 8, 9):
            stack.operations[i].rotate(np.pi, [0, 1, 0])

        return stack

    def get_cylinder(self) -> Cylinder:
        return Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])

    def get_frustum(self) -> Frustum:
        return Frustum([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.3)

    def get_box(self) -> Box:
        return Box([0, 0, 0], [1, 1, 1])

    def setUp(self):
        self.mesh = Mesh()


class GraderTests(AutogradeTestsBase):
    def test_fixed_count_cylinder(self):
        cylinder = self.get_cylinder()
        self.mesh.add(cylinder)
        self.mesh.assemble()

        grader = FixedCountGrader(self.mesh, 10)
        grader.grade()

        for block in self.mesh.blocks:
            for axis in block.axes:
                self.assertEqual(axis.count, 10)

    def test_simple_grader_stack(self):
        stack = self.get_stack()
        self.mesh.add(stack)
        self.mesh.assemble()

        grader = SimpleGrader(self.mesh, 0.05)
        grader.grade()

        for block in self.mesh.blocks:
            for axis in block.axes:
                self.assertEqual(axis.count, 5)

    def test_smooth_cylinder(self):
        self.mesh.add(self.get_cylinder())
        self.mesh.assemble()

        grader = SmoothGrader(self.mesh, 0.025)
        grader.grade()

        # make sure all blocks are defined
        for block in self.mesh.blocks:
            self.assertTrue(block.is_defined)


class AdvancedAutogradeTests(unittest.TestCase):
    def setUp(self):
        self.mesh = Mesh()

    def get_grid(self, badness: float = 0.8):
        # 6 blocks of weird shape, taken from the autograder examples
        # but with adjustable 'badness' (amount of displacement)
        # to test border cases
        base = cb.Grid([0, 0, 0], [3, 2, 0], 3, 2)
        shape = cb.ExtrudedShape(base, 1)

        # turn one block around
        shape.grid[1][0].rotate(np.pi, [0, 0, 1])
        self.mesh.add(shape)

        # move some points to get a mesh with uneven blocks
        self.mesh.assemble()
        finder = cb.GeometricFinder(self.mesh)

        move_points = [[0, 1, 1], [2, 1, 1], [3, 1, 1]]

        for point in move_points:
            vertex = next(iter(finder.find_in_sphere(point)))
            vertex.translate([0, badness, 0])

    def test_squeezed_chop(self):
        self.get_grid()

        grader = SmoothGrader(self.mesh, 0.05)
        grader.grade()

        self.assertIsNone(self.mesh.blocks[5].axes[1].wires[1].grading.chops[0].total_expansion)
