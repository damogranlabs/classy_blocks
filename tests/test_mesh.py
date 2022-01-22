import unittest
import os

import numpy as np

from classy_blocks.classes.primitives import Vertex, Edge
from classy_blocks.classes.grading import Grading
from classy_blocks.classes.mesh import Mesh
from classy_blocks.util import constants

from tests.test_block import FixturedTestCase

class TestMesh(FixturedTestCase):
    def test_find_vertex(self):
        """ an existing object must be returned when adding a vertex at an existing location """
        v = Vertex([0, 0, 0])
        v.index = 0

        mesh = Mesh()
        mesh.vertices.append(v)
        
        self.assertEqual(v, mesh.find_vertex(Vertex([0, 0, 0])))
        self.assertIsNone(mesh.find_vertex(Vertex([1, 1, 1])))

    def test_find_edge(self):
        """ edges between the same vertices must not be duplicated """
        mesh = Mesh()

        v1 = Vertex([0, 0, 0])
        v1.mesh_index = 0
        mesh.vertices.append(v1)

        v2 = Vertex([1, 0, 0])
        v2.mesh_index = 1
        mesh.vertices.append(v2)

        e = Edge(0, 1, [0.5, 0.1, 0])
        e.vertex_1 = v1
        e.vertex_2 = v2
        mesh.edges.append(e)

        self.assertEqual(e, mesh.find_edge(v1, v2))

        # add a new vertex: find_edge shouldn't find anything
        # related to this vertex
        v3 = Vertex([1, 1, 0])
        v3.mesh_index = 2
        mesh.vertices.append(v3)
        
        self.assertIsNone(mesh.find_edge(v2, v3))
        self.assertIsNone(mesh.find_edge(v1, v3))

    def test_prepare_data(self):
        """ a functional test on mesh.prepare_data() """
        self.mesh.prepare_data()

        # 8 out of 24 vertices are shared between blocks and must not be duplicated
        self.assertEqual(len(self.mesh.vertices), 16)

        # check that it's the same vertex
        self.assertEqual(self.block_0.vertices[1], self.block_1.vertices[0])
        self.assertEqual(self.block_0.vertices[2], self.block_1.vertices[3])
        self.assertEqual(self.block_0.vertices[5], self.block_1.vertices[4])
        self.assertEqual(self.block_0.vertices[6], self.block_1.vertices[7])

        # only 2 out of 4 edges should be added (one is duplicated, one invalid)
        self.assertEqual(len(self.mesh.edges), 2)

    def test_patches(self):
        """ check patch output """
        self.mesh.prepare_data()

        self.assertDictEqual(
            self.mesh.patches,
            {
                'inlet': ['(4 0 3 7)'],
                'outlet': ['(15 14 12 13)'],
                'walls': [
                    '(0 1 2 3)', '(4 5 6 7)', '(4 5 1 0)', '(7 6 2 3)', # block_0
                    '(1 8 9 2)', '(5 10 11 6)', '(10 8 9 11)', '(5 10 8 1)', # block_1
                    '(2 9 12 13)', '(6 11 14 15)', '(6 2 13 15)', '(11 9 12 14)' # block_2
                ],
            }
        )

    def test_find_neighbour_success(self):
        """ block_2 must copy block_1's cell count and grading on axis 0 and 1"""
        self.mesh.prepare_data()

        self.assertTrue(self.mesh.copy_grading(self.block_2, 0))
        self.assertTrue(self.mesh.copy_grading(self.block_2, 1))
        
    def test_find_neighbour_fail(self):
        """ block_2 cannot copy cell count and grading from block_1 on axis 2 """
        self.block_1.grading = [Grading(), Grading(), Grading()]
        self.block_1.deferred_gradings = []
        
        self.assertRaises(Exception, self.mesh.prepare_data)


if __name__ == '__main__':
    unittest.main()