import unittest

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.operations.box import Box
from classy_blocks.optimize.mapper import Mapper


class MapperTests(unittest.TestCase):
    def test_single_face(self):
        mapper = Mapper()
        mapper.add(Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]))

        self.assertListEqual(mapper.indexes, [[0, 1, 2, 3]])

    def test_two_faces(self):
        mapper = Mapper()
        face = Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        mapper.add(face)
        mapper.add(face.copy().translate([1, 0, 0]))

        self.assertListEqual(mapper.indexes, [[0, 1, 2, 3], [1, 4, 5, 2]])

    def test_single_operation(self):
        mapper = Mapper()
        box = Box([0, 0, 0], [1, 1, 1])

        mapper.add(box)

        self.assertListEqual(mapper.indexes, [[0, 1, 2, 3, 4, 5, 6, 7]])

    def test_two_operations(self):
        mapper = Mapper()
        box = Box([0, 0, 0], [1, 1, 1])

        mapper.add(box)
        mapper.add(box.copy().translate([1, 0, 0]))

        self.assertListEqual(mapper.indexes, [[0, 1, 2, 3, 4, 5, 6, 7], [1, 8, 9, 2, 5, 10, 11, 6]])
