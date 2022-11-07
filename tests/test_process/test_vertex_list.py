import unittest

from classy_blocks.define.block import Block
from classy_blocks.define.primitives import Vertex
from classy_blocks.util import functions as f
from classy_blocks.util import constants

from classy_blocks.process.lists.vertices import VertexList

from tests.fixtures import FixturedTestCase

class VertexListTests(FixturedTestCase):
    def test_collect_vertices_single(self):
        """Collect vertices from a single block"""
        vertices = VertexList()
        vertices.collect([self.block_0])

        self.assertEqual(len(vertices), 8)

    def test_collect_vertices_multiple(self):
        """Collect vertices from two touching blocks"""
        vertices = VertexList()
        vertices.collect([self.block_0, self.block_1])

        self.assertEqual(len(vertices), 12)
        
    def test_collect_vertices_indexes(self):
        """Check that the correct vertices are assigned to block
        on collect()"""
        vertices = VertexList()
        vertices.collect([self.block_0, self.block_1])

        # the second block should reuse some vertices
        first_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        second_indexes = [1, 8, 9, 2, 5, 10, 11, 6]

        self.assertListEqual([v.mesh_index for v in self.block_0.vertices], first_indexes)
        self.assertListEqual([v.mesh_index for v in self.block_1.vertices], second_indexes)

    def test_find(self):
        vertices = VertexList()
        vertices.collect([self.block_0, self.block_1])

        displacement = constants.tol/10

        for i, vertex in enumerate(vertices):
            # we're searching for this point
            point = vertex.point

            # but slightly displaced (well within tolerances)
            point = point + f.vector(displacement, displacement, displacement)

            self.assertEqual(vertices.find(point).mesh_index, i)

    def test_find_fail(self):
        vertices = VertexList()
        vertices.collect([self.block_0])

        self.assertIsNone(vertices.find(f.vector(999, 999, 999)))

    def test_output(self):
        vertices = VertexList()
        vertices.collect([self.block_0])
        self.maxDiff = None

        expected_output = "vertices\n(\n" + \
            "\t(0.00000000 0.00000000 0.00000000) // 0\n" + \
            "\t(1.00000000 0.00000000 0.00000000) // 1\n" + \
            "\t(1.00000000 1.00000000 0.00000000) // 2\n" + \
            "\t(0.00000000 1.00000000 0.00000000) // 3\n" + \
            "\t(0.00000000 0.00000000 1.00000000) // 4\n" + \
            "\t(1.00000000 0.00000000 1.00000000) // 5\n" + \
            "\t(1.00000000 1.00000000 1.00000000) // 6\n" + \
            "\t(0.00000000 1.00000000 1.00000000) // 7\n" + \
            ");\n"

        self.assertEqual(vertices.output(), expected_output)