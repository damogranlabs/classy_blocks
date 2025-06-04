from parameterized.parameterized import parameterized

from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.mesh import Mesh
from classy_blocks.modify.find.geometric import GeometricFinder
from classy_blocks.modify.find.shape import RoundSolidFinder
from tests.fixtures.block import BlockTestCase


class GeometricFinderTests(BlockTestCase):
    def setUp(self):
        super().setUp()

        self.mesh = Mesh()

        loft = self.make_loft(0)
        self.mesh.add(loft)
        self.mesh.assemble()

        self.finder = GeometricFinder(self.mesh)

    def test_by_position_close(self):
        """Find one exact vertex"""
        found_vertices = self.finder.find_in_sphere([0, 0, 0])

        self.assertSetEqual(found_vertices, {self.mesh.vertices[0]})

    def test_by_position_close_count(self):
        """Find one exact vertex"""
        found_vertices = self.finder.find_in_sphere([0, 0, 0])

        self.assertEqual(len(found_vertices), 1)

    def test_by_position_far(self):
        """Find vertices on cube"""
        found_vertices = self.finder.find_in_sphere([0, 0, 0], 1.0001)
        # the loft #0 is a cube
        self.assertEqual(len(found_vertices), 4)

    def test_by_position_all(self):
        """Find all vertices of this mesh"""
        found_vertices = self.finder.find_in_sphere([0, 0, 0], 2)

        self.assertEqual(len(found_vertices), 8)

    def test_on_plane_bottom(self):
        found_vertices = self.finder.find_on_plane([0, 0, 0], [0, 0, 1])

        self.assertEqual(len(found_vertices), 4)

    def test_on_plane_top(self):
        found_vertices = self.finder.find_on_plane([0, 0, 1], [0, 0, 1])

        self.assertEqual(len(found_vertices), 4)


class RoundSolidShapeFinderTests(BlockTestCase):
    def setUp(self):
        super().setUp()

        self.mesh = Mesh()
        self.cylinder = Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])
        self.mesh.add(self.cylinder)
        self.mesh.assemble()

        self.finder = RoundSolidFinder(self.mesh, self.cylinder)

    @parameterized.expand(((True,), (False,)))
    def test_core_count(self, end_face):
        """Number of vertices on core"""
        vertices = self.finder.find_core(end_face)

        self.assertEqual(len(vertices), 9)

    @parameterized.expand(((True,), (False,)))
    def test_shell_count(self, end_face):
        """Number of vertices on shell"""
        vertices = self.finder.find_shell(end_face)

        self.assertEqual(len(vertices), 8)
