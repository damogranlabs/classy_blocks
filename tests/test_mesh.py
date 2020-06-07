import unittest
import os

import numpy as np
from classes.primitives import Vertex, Edge
from classes.mesh import Mesh

from util import constants

from tests.test_block import FixturedTestCase

class TestMesh(FixturedTestCase):
    def test_find_vertex(self):
        # an existing object must be returned when adding
        # a vertex at an existing location
        v = Vertex([0, 0, 0])
        v.index = 0

        mesh = Mesh()
        mesh.vertices.append(v)
        
        self.assertEqual(v, mesh.find_vertex(Vertex([0, 0, 0])))
        self.assertIsNone(mesh.find_vertex(Vertex([1, 1, 1])))

    def test_find_edge(self):
        # do not duplicate edges between the same vertices
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
        self.mesh.prepare_data()

        # 4 out of 16 vertices are shared between blocks and must not be duplicated
        self.assertEqual(len(self.mesh.vertices), 12)

        # check that it's the same vertex
        self.assertEqual(self.block_1.vertices[4], self.block_2.vertices[0])
        self.assertEqual(self.block_1.vertices[5], self.block_2.vertices[1])
        self.assertEqual(self.block_1.vertices[6], self.block_2.vertices[2])
        self.assertEqual(self.block_1.vertices[7], self.block_2.vertices[3])

        # only 4 out of 6 edges should be added (one is duplicated, one invalid)
        self.assertEqual(len(self.mesh.edges), 4)

    """def test_patches(self):
        self.assertDictEqual(
            self.mesh.patches,
            {
                'outlet': ['(4 5 6 7)'],
                'walls': ['(4 0 3 7)', '(5 1 2 6)', '(4 5 1 0)', '(7 6 2 3)'],
                'inlet': ['(0 1 2 3)']
            }
        )"""


if __name__ == '__main__':
    unittest.main()