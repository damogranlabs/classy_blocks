import unittest

import numpy as np
from classes.primitives import Vertex, Edge
from classes.mesh import Mesh

from util import constants

from tests.fixtures import FixturedTestCase

class TestBlock(FixturedTestCase):
    ###
    ### Block tests
    ###
    def test_block_definition(self):
        self.mesh.prepare_data()

        self.assertEqual(
            str(self.block_1),
            "hex  ( 0 1 2 3 4 5 6 7 )  (10 10 10)  simpleGrading (1 1 1) // 0 Test"
        )
    
    def test_face_format(self):
        self.mesh.prepare_data()

        self.assertEqual(
            self.block_1.format_face('bottom'),
            "(0 1 2 3)"
        )
    
    def test_patches(self):
        self.mesh.prepare_data()

        self.assertListEqual(self.block_1.patches['inlet'], ['bottom'])
        self.assertListEqual(self.block_1.patches['outlet'], ['top'])
        self.assertListEqual(self.block_1.patches['walls'], ['left', 'right', 'front', 'back'])

    def test_faces(self):
        self.mesh.prepare_data()

        self.assertListEqual(
            self.block_1.get_faces('walls'),
            ['(4 0 3 7)', '(5 1 2 6)', '(4 5 1 0)', '(7 6 2 3)']
        )

    def test_block_size(self):
        for i in range(3):
            self.assertEqual(self.block_1.size[i], 1)

    def test_cell_size(self):
        axis = 0
        test_cell_size = 0.005
        n = self.block_1.n_cells[axis]

        # set the desired cell size
        self.block_1.set_cell_size(axis, test_cell_size)
        # get block grading
        g = self.block_1.grading[axis]

        # check that the sum of all elements, graded, is block size
        cell_sizes = [test_cell_size]
        block_size = 0

        for _ in range(n):
            s = cell_sizes[-1]*(g**(1/n))
            cell_sizes.append(s)
            block_size += s

        self.assertAlmostEqual(block_size, self.block_1.size[axis])

        # also check that ratio of first to last cell size is what's calculated
        self.assertAlmostEqual(cell_sizes[-1]/cell_sizes[0], g, delta=constants.tol)

if __name__ == '__main__':
    unittest.main()