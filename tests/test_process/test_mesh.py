from tests.fixtures import FixturedTestCase


class TestMesh(FixturedTestCase):
    def test_prepare(self):
        """a functional test on mesh.write()"""
        self.prepare()

        # 8 out of 24 vertices are shared between blocks and must not be duplicated
        self.assertEqual(len(self.mesh.vertices), 16)

        # check that it's the same vertex
        self.assertEqual(self.block_0.vertices[1], self.block_1.vertices[0])
        self.assertEqual(self.block_0.vertices[2], self.block_1.vertices[3])
        self.assertEqual(self.block_0.vertices[5], self.block_1.vertices[4])
        self.assertEqual(self.block_0.vertices[6], self.block_1.vertices[7])

        # only 2 out of 4 edges should be added (one is duplicated, one invalid)
        self.assertEqual(len(self.mesh.edges), 2)
