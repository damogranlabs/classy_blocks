import unittest

import numpy as np

from classy_blocks.construct.flat.sketches.grid import Grid
from classy_blocks.construct.operations.box import Box
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.construct.shapes.frustum import Frustum
from classy_blocks.construct.stack import ExtrudedStack
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
