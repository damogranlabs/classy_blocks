import numpy as np

from classy_blocks.base.exceptions import VertexNotFoundError
from classy_blocks.items.vertex import Vertex
from classy_blocks.util import constants
from classy_blocks.util import functions as f

from tests.fixtures.data import DataTestCase

class VertexTests(DataTestCase):
    def setUp(self):
        Vertex.registry = []
    
    def test_assert_3d(self):
        """Raise an exception if the point is not in 3D space"""
        with self.assertRaises(AssertionError):
            _ = Vertex([0, 0])

    def test_translate_int(self):
        """Vertex translation with integer delta"""
        delta = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            Vertex([0, 0, 0]).translate(delta).pos, delta
        )
    
    def test_translate_float(self):
        """Vertex translation with float delta"""
        coords = [0.0, 0.0, 0.0]
        delta = [1.0, 0.0, 0.0]

        np.testing.assert_array_almost_equal(
            Vertex(coords).translate(delta).pos, delta
        )
    
    def test_rotate(self):
        """Rotate a point"""
        coords = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            Vertex(coords).rotate(np.pi/2, [0, 0, 1], [0, 0, 0]).pos,
            [0, 1, 0]
        )

    def test_scale(self):
        """Scale a point"""
        coords = [1, 0, 0]

        np.testing.assert_array_almost_equal(
            Vertex(coords).scale(2, [0, 0, 0]).pos,
            [2, 0, 0]
        )

    def test_inequal(self):
        """The __eq__ method returns False"""
        point_1 = Vertex([0, 0, 0])
        point_2 = Vertex([0, 0, 0 + 2*constants.tol])

        self.assertFalse(point_1 == point_2)

    def test_different(self):
        """Create two different vertices"""
        # expect the list of vertices to have two elements
        vertex_1 = Vertex([0, 0, 0])
        vertex_2 = Vertex([1, 0, 0])

        self.assertEqual(vertex_1.index, 0)
        self.assertEqual(vertex_2.index, 1)

    def test_equal(self):
        """Create two vertices at the same position and check that the
        same object is returned"""
        vertex_1 = Vertex([0, 0, 0])
        vertex_2 = Vertex([0, 0, 0])

        self.assertEqual(id(vertex_1), id(vertex_2))

    def test_almost_equal(self):
        """Same as test_equal but with a slight (numerical) offset"""
        vertex_1 = Vertex([0, 0, 0])
        vertex_2 = Vertex([0, 0, constants.tol/2])

        self.assertEqual(id(vertex_1), id(vertex_2))

    def test_collect_vertices_single_block(self):
        """Collect vertices from a single block"""
        vertices = [Vertex(p) for p in self.get_single_data(0).points]

        self.assertEqual(len(vertices), 8)
        self.assertEqual(len(Vertex.registry), 8)

    def test_collect_vertices_multiple(self):
        """Collect vertices from two touching blocks"""
        _ = [Vertex(p) for p in self.get_single_data(0).points]
        _ = [Vertex(p) for p in self.get_single_data(1).points]

        self.assertEqual(len(Vertex.registry), 12)
        
    def test_collect_vertices_indexes(self):
        """Check that the correct vertices are assigned to block
        on collect()"""
        _ = [Vertex(p) for p in self.get_single_data(0).points]
        _ = [Vertex(p) for p in self.get_single_data(1).points]

        # the second block should reuse some vertices
        first_indexes = [0, 1, 2, 3, 4, 5, 6, 7]
        second_indexes = [1, 8, 9, 2, 5, 10, 11, 6]

        # compare collected vertices by position
        all_points = np.array([v.pos for v in Vertex.registry])
        first_points = np.take(all_points, first_indexes, axis=0)

        second_points = np.take(all_points, second_indexes, axis=0)

        np.testing.assert_array_equal(first_points, self.get_single_data(0).points)
        np.testing.assert_array_equal(second_points, self.get_single_data(1).points)

    def test_find_success(self):
        """Match vertex positions and expected indexes"""
        _ = [Vertex(p) for p in self.get_single_data(0).points]
        _ = [Vertex(p) for p in self.get_single_data(1).points]

        self.assertEqual(len(Vertex.registry), 12)

        displacement = constants.tol/10

        for i, vertex in enumerate(Vertex.registry):
            # we're searching for this point
            # but slightly displaced (well within tolerances)
            point = vertex.pos + f.vector(displacement, displacement, displacement)

            self.assertEqual(Vertex.find(point).index, i)

    def test_find_fail(self):
        """Raise a RuntimeError when no Vertex exists at a given position"""
        _ = [Vertex(p) for p in self.get_single_data(0).points]

        with self.assertRaises(VertexNotFoundError):
            Vertex.find(f.vector(999, 999, 999))
