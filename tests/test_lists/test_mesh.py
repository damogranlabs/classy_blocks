from tests.fixtures.data import DataTestCase

from classy_blocks.mesh import Mesh


class TestMesh(DataTestCase):
    """Mesh() object tests"""
    def setUp(self):
        super().setUp()
        self.mesh = Mesh()

    def test_add_operation(self):
        """Add a block to the Mesh()"""
        self.mesh.add(self.get_single_data(0))

        self.assertEqual(len(self.mesh.block_list.blocks), 1)
    
    def test_shared_edge(self):
        """Blocks 0 and 1 should have the same Edge object"""
        self.mesh.add(self.get_single_data(0))
        self.mesh.add(self.get_single_data(1))

        self.assertEqual(
            id(self.mesh.block_list.blocks[0].frame.edges[1]),
            id(self.mesh.block_list.blocks[1].frame.edges[0])
        )

    # def test_prepare(self):
    #     """a functional test on mesh.write()"""
    #     self.mesh.prepare()

    #     # 8 out of 24 vertices are shared between blocks and must not be duplicated
    #     self.assertEqual(len(self.mesh.vertices), 16)

    #     # check that it's the same vertex
    #     self.assertEqual(self.block_0.vertices[1], self.block_1.vertices[0])
    #     self.assertEqual(self.block_0.vertices[2], self.block_1.vertices[3])
    #     self.assertEqual(self.block_0.vertices[5], self.block_1.vertices[4])
    #     self.assertEqual(self.block_0.vertices[6], self.block_1.vertices[7])

    #     # only 2 out of 4 edges should be added (one is duplicated, one invalid)
    #     self.assertEqual(len(self.mesh.edges), 2)
