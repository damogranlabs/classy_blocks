from typing import get_args

from classy_blocks.construct.operations.box import Box
from classy_blocks.mesh import Mesh
from classy_blocks.types import DirectionType
from tests.fixtures.data import DataTestCase


class MeshTestCase(DataTestCase):
    """A test case where all blocks are already in the mesh after setUp"""

    def make_box(self, index):
        data = self.get_single_data(index)

        box = Box(data.points[0], data.points[6])

        for i in get_args(DirectionType):
            box.chop(i, count=10)

        return box

    def setUp(self):
        super().setUp()

        self.mesh = Mesh()
        self.boxes = []

        for i in range(3):
            box = self.make_box(i)
            self.mesh.add(box)

        self.mesh.assemble()
