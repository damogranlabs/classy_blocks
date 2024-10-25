import unittest

from classy_blocks.construct.flat.sketches.grid import Grid
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.construct.shapes.frustum import Frustum
from classy_blocks.construct.stack import ExtrudedStack
from classy_blocks.grading.autograding.grader import HighReGrader
from classy_blocks.grading.autograding.params import HighReChopParams
from classy_blocks.mesh import Mesh


class AutogradeTestsBase(unittest.TestCase):
    def get_stack(self) -> ExtrudedStack:
        # create a simple 3x3 grid for easy navigation
        base = Grid([0, 0, 0], [1, 1, 0], 3, 3)
        return ExtrudedStack(base, 1, 3)

    def get_cylinder(self) -> Cylinder:
        return Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])

    def get_frustum(self) -> Frustum:
        return Frustum([0, 0, 0], [1, 0, 0], [0, 1, 0], 0.3)

    def setUp(self):
        self.mesh = Mesh()


class GraderTests(AutogradeTestsBase):
    def test_highre_cylinder(self):
        self.mesh.add(self.get_cylinder())
        self.mesh.assemble()
        # TODO: Hack! Un-hack!
        self.mesh.block_list.update()

        params = HighReChopParams(0.025)
        grader = HighReGrader(self.mesh, params)

        grader.grade()
