from classy_blocks.mesh import Mesh
from classy_blocks.modify.find.vertex import VertexFinder
from tests.fixtures.block import BlockTestCase


class VertexFinderTests(BlockTestCase):
    def setUp(self):
        super().setUp()

        self.mesh = Mesh()

        loft = self.make_loft(0)
        self.mesh.add(loft)
        self.mesh.assemble()

        self.finder = VertexFinder(self.mesh)

    def test_by_position_close(self):
        """Find one exact vertex"""
        found_vertices = self.finder.by_position([0, 0, 0])

        self.assertEqual(found_vertices[0], self.mesh.vertex_list.vertices[0])

    def test_by_position_close_count(self):
        """Find one exact vertex"""
        found_vertices = self.finder.by_position([0, 0, 0])

        self.assertEqual(len(found_vertices), 1)

    def test_by_position_far(self):
        """Find vertices on cube"""
        found_vertices = self.finder.by_position([0, 0, 0], 1.0001)
        # the loft #0 is a cube
        self.assertEqual(len(found_vertices), 4)

    def test_by_position_all(self):
        """Find all vertices of this mesh"""
        found_vertices = self.finder.by_position([0, 0, 0], 2)

        self.assertEqual(len(found_vertices), 8)
