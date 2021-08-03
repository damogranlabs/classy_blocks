import unittest

import numpy as np
from classy_blocks.classes.primitives import Vertex, Edge
from classy_blocks.classes.block import Block
from classy_blocks.classes.mesh import Mesh

from classy_blocks.util import constants

class FixturedTestCase(unittest.TestCase):
    """ common setUp for block and mesh tests """
    def setUp(self):
        # a test mesh
        self.block_1_points = [
            [0, 0, 0], # 0
            [1, 0, 0], # 1
            [1, 1, 0], # 2
            [0, 1, 0], # 3

            [0, 0, 1], # 4
            [1, 0, 1], # 5
            [1, 1, 1], # 6
            [0, 1, 1], # 7
        ]

        self.block_2_points = [
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],

            [0, 0, 2],
            [1, 0, 2],
            [1, 1, 2],
            [0, 1, 2],
        ]

        self.block_1_edges = [
            Edge(0, 1, [0.5, -0.25, 0]), # arc edge
            Edge(1, 2, [[1.2, 0.25, 0], [1.3, 0.5, 0], [1.3, 0.75, 0]]), # spline edge

            Edge(4, 5, [0.5, -0.1, 1]),
            Edge(6, 7, [[0.7, 1.1, 1], [0.3, 1.1, 1]])
        ]

        # additional edge that must not be included (already in block_1_edges)
        self.block_2_edges = [
            Edge(0, 1, [0.5, -0.1, 1]),
            Edge(4, 5, [0.5, 0, 2]) # collinear point between vertex 4 and 5
        ]

        # the most low-level way of creating a block is from 'raw' points
        self.block_1 = Block.create_from_points(self.block_1_points, self.block_1_edges)
        self.block_1.n_cells = [5, 10, 15]

        self.block_2 = Block.create_from_points(self.block_2_points, self.block_2_edges)
        self.block_2.n_cells[2] = 20

        # other block data
        self.block_1.description = "Test"
        self.block_1.set_patch('bottom', 'inlet')
        self.block_1.set_patch('top', 'outlet')
        self.block_1.set_patch(['left', 'right', 'front', 'back'], 'walls')

        self.mesh = Mesh()
        self.mesh.add_block(self.block_1)
        self.mesh.add_block(self.block_2)

