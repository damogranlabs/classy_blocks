from tests.fixtures.block import BlockTestCase

from classy_blocks.items.vertex import Vertex
from classy_blocks.items.side import Side


class SideTests(BlockTestCase):
    def test_create_fail(self):
        """Attempt to pass only side vertices instead of the whole set"""
        vertices = [Vertex([0, 0, 0], 0), Vertex([1, 0, 0], 1)]
        with self.assertRaises(AssertionError):
            Side("left", vertices)

    def test_create_success(self):
        """Create a Side object and test its contents"""
        vertices = self.make_vertices(0)
        side = Side("bottom", vertices)

        self.assertListEqual(side.vertices, [vertices[i] for i in (0, 1, 2, 3)])

    def test_description(self):
        """String output"""
        vertices = self.make_vertices(0)
        side = Side("bottom", vertices)

        self.assertEqual(side.description, "(0 1 2 3)")

    def test_equal(self):
        """Two coincident sides from different blocks are equal"""
        side_1 = Side("right", self.make_vertices(0))
        side_2 = Side("left", self.make_vertices(1))

        self.assertTrue(side_1 == side_2)
