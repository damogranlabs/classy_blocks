import numpy as np

from classy_blocks.util import functions as f
from classy_blocks.util import constants

from classy_blocks.base.exceptions import VertexNotFoundError
from classy_blocks.lists.vertex_list import VertexList, DuplicatedEntry

from tests.fixtures.data import DataTestCase

class VertexListTests(DataTestCase):
    def setUp(self):
        super().setUp()
        self.blocks = self.get_all_data()
        self.vlist = VertexList()

    def add_all(self, points):
        vertices = []
        for p in points:
            vertices.append(self.vlist.add(p))
        
        return vertices

    def test_collect_vertices_single(self):
        """Collect vertices from a single block"""
        vertices = self.add_all(self.blocks[0].points)

        self.assertEqual(len(self.vlist.vertices), 8)
        self.assertEqual(len(vertices), 8)

    def test_collect_vertices_multiple(self):
        """Collect vertices from two touching blocks"""
        self.add_all(self.blocks[0].points)
        self.add_all(self.blocks[1].points)

        self.assertEqual(len(self.vlist.vertices), 12)
        
    def test_collect_vertices_indexes(self):
        """Check that the correct vertices are assigned to block
        on collect()"""
        self.add_all(self.blocks[0].points)
        self.add_all(self.blocks[1].points)

        # the second block should reuse some vertices
        first_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        second_indexes = [1, 8, 9, 2, 5, 10, 11, 6]

        # compare collected vertices by position
        all_points = np.array([v.pos for v in self.vlist.vertices])
        first_points = np.take(all_points, first_indexes, axis=0)
        second_points = np.take(all_points, second_indexes, axis=0)

        np.testing.assert_array_equal(first_points, self.blocks[0].points)
        np.testing.assert_array_equal(second_points, self.blocks[1].points)

    def test_find_success(self):
        """Find an existing vertex at specified point"""
        self.add_all(self.blocks[0].points)
        self.add_all(self.blocks[1].points)

        displacement = constants.TOL/10

        for i, vertex in enumerate(self.vlist.vertices):
            # we're searching for this point
            # but slightly displaced (well within tolerances)
            point = vertex.pos + f.vector(displacement, displacement, displacement)

            self.assertEqual(self.vlist.find(point).index, i)

    def test_find_fail(self):
        """Raise an error when no vertex was found at specified point"""
        self.add_all(self.blocks[0].points)

        with self.assertRaises(VertexNotFoundError):
            self.vlist.find(f.vector(999, 999, 999))

    def test_find_duplicated_success(self):
        """An existing vertex at specified point and slave patch was found"""
        self.add_all(self.blocks[0].points)
        master_vertex = self.vlist.vertices[0]
        self.vlist.duplicated = [DuplicatedEntry(master_vertex, ['terrain'])]

        self.assertEqual(
            self.vlist.find(master_vertex.pos, ['terrain']),
            self.vlist.vertices[0]
        )

    def test_find_duplicated_fail(self):
        """Raise a VertexNotFoundError when no vertices are on the specified slave patch"""
        self.add_all(self.blocks[0].points)

        with self.assertRaises(VertexNotFoundError):
            _ = self.vlist.find(self.vlist.vertices[0].pos, ['terrain'])

    def test_add_slave_single(self):
        """Add a vertex on slave patch; must be duplicated"""
        self.add_all(self.blocks[0].points)

        new_vertex = self.vlist.add(self.vlist.vertices[0].pos, ['terrain'])

        self.assertEqual(len(self.vlist.vertices), 9)
        self.assertNotEqual(self.vlist.vertices[0], new_vertex)
    
    def test_add_slave_multiple(self):
        """Add a vertex on slave patch multiple times;
        must be duplicated only once"""
        self.add_all(self.blocks[0].points)

        self.vlist.add(self.vlist.vertices[0].pos, ['terrain'])
        self.vlist.add(self.vlist.vertices[0].pos, ['terrain'])
        self.vlist.add(self.vlist.vertices[0].pos, ['terrain'])

        self.assertEqual(len(self.vlist.vertices), 9)
