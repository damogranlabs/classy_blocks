import numpy as np

from classy_blocks.util import functions as f
from classy_blocks.util import constants

from classy_blocks.data.point import Point
from classy_blocks.lists.vertex_list import VertexList

from tests.fixtures.data import DataTestCase

class VertexListTests(DataTestCase):
    def setUp(self):
        super().setUp()
        self.blocks = self.get_all_data()
        self.vlist = VertexList()

    def test_collect_vertices_single(self):
        """Collect vertices from a single block"""
        vertices = self.vlist.add(self.blocks[0].points)

        self.assertEqual(len(self.vlist.vertices), 8)
        self.assertEqual(len(vertices), 8)

    def test_collect_vertices_multiple(self):
        """Collect vertices from two touching blocks"""
        self.vlist.add(self.blocks[0].points)
        self.vlist.add(self.blocks[1].points)

        self.assertEqual(len(self.vlist.vertices), 12)
        
    def test_collect_vertices_indexes(self):
        """Check that the correct vertices are assigned to block
        on collect()"""
        self.vlist.add(self.blocks[0].points)
        self.vlist.add(self.blocks[1].points)

        # the second block should reuse some vertices
        first_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        second_indexes = [1, 8, 9, 2, 5, 10, 11, 6]

        # compare collected vertices by position
        all_points = np.array([v.pos for v in self.vlist.vertices])
        first_points = np.take(all_points, first_indexes, axis=0)
        first_points = [Point(p) for p in first_points]

        second_points = np.take(all_points, second_indexes, axis=0)
        second_points = [Point(p) for p in second_points]

        self.assertListEqual(first_points, self.blocks[0].points)
        self.assertListEqual(second_points, self.blocks[1].points)

    def test_find(self):
        self.vlist.add(self.blocks[0].points)
        self.vlist.add(self.blocks[1].points)

        displacement = constants.TOL/10

        for i, vertex in enumerate(self.vlist.vertices):
            # we're searching for this point
            # but slightly displaced (well within tolerances)
            point = vertex.pos + f.vector(displacement, displacement, displacement)

            self.assertEqual(self.vlist.find(point).index, i)

    def test_find_fail(self):
        self.vlist.add(self.blocks[0].points)

        with self.assertRaises(RuntimeError):
            self.vlist.find(f.vector(999, 999, 999))

    # def test_output(self):
    #     vertices = VertexList()
    #     vertices.collect([self.block_0], [])
    #     self.maxDiff = None

    #     expected_output = "vertices\n(\n" + \
    #         "\t(0.00000000 0.00000000 0.00000000) // 0\n" + \
    #         "\t(1.00000000 0.00000000 0.00000000) // 1\n" + \
    #         "\t(1.00000000 1.00000000 0.00000000) // 2\n" + \
    #         "\t(0.00000000 1.00000000 0.00000000) // 3\n" + \
    #         "\t(0.00000000 0.00000000 1.00000000) // 4\n" + \
    #         "\t(1.00000000 0.00000000 1.00000000) // 5\n" + \
    #         "\t(1.00000000 1.00000000 1.00000000) // 6\n" + \
    #         "\t(0.00000000 1.00000000 1.00000000) // 7\n" + \
    #         ");\n\n"

    #     self.assertEqual(vertices.output(), expected_output)
