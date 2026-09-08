import unittest

from classy_blocks.construct.shapes.mapped import MappedShape
from classy_blocks.mesh import Mesh
from tests.fixtures import data


class MappedShapeTests(unittest.TestCase):
    def setUp(self):
        self.points = data.fl + data.cl
        self.hexas = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [1, 8, 9, 2, 5, 10, 11, 6],
            [2, 9, 12, 13, 6, 11, 14, 15],
        ]

    def test_simple(self):
        # a simple 3-block shape
        shape = MappedShape(self.points, self.hexas)
        mesh = Mesh()
        mesh.add(shape)

        mesh.assemble()

        self.assertEqual(len(mesh.dump.blocks), 3)
