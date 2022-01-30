import unittest

import numpy as np

from classy_blocks.classes.block import Block
from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.grading import Grading
from classy_blocks.util import constants

from tests.fixtures import FixturedTestCase, ExecutedTestsBase

class TestBlock(FixturedTestCase, ExecutedTestsBase):
    ###
    ### Block tests
    ###
    def test_block_definition(self):
        """ the correct output for blockMeshDict """
        self.mesh.prepare_data()

        self.assertEqual(
            str(self.block_0),
            "hex  ( 0 1 2 3 4 5 6 7 )  (6 6 7)  simpleGrading (1 1 1) // 0 Test"
        )

    def test_is_grading_defined(self):
        self.mesh.prepare_data()
        self.assertTrue(self.block_1.is_grading_defined)

        self.block_1.grading[0] = Grading()
        self.assertFalse(self.block_1.is_grading_defined)
   
    def test_face_format(self):
        """ the correct face format for blockMeshDict """
        self.mesh.prepare_data()

        self.assertEqual(
            self.block_0.format_face('bottom'),
            "(0 1 2 3)"
        )
    
    def test_patches(self):
        """ patch naming/positions """
        self.mesh.prepare_data()

        self.assertListEqual(self.block_0.patches['inlet'], ['left'])
        self.assertListEqual(self.block_2.patches['outlet'], ['back'])

        self.assertListEqual(
            self.block_0.patches['walls'], ['bottom', 'top', 'front', 'back'])

        self.assertListEqual(self.block_1.patches['walls'], ['bottom', 'top', 'right', 'front'])

        self.assertListEqual(self.block_2.patches['outlet'], ['back'])
        self.assertListEqual(self.block_2.patches['walls'], ['bottom', 'top', 'left', 'right'])

    def test_faces(self):
        """ definitions of faces around the block """
        self.mesh.prepare_data()

        self.assertListEqual(
            self.block_0.get_faces('walls'),
            [[0, 1, 2, 3], [4, 5, 6, 7], [4, 5, 1, 0], [7, 6, 2, 3]]
        )

    def test_straight_block_size(self):
        """ length of a straight block edge """
        self.mesh.prepare_data()

        self.assertEqual(self.block_1.get_size(2), 1)
    
    def test_arc_block_size(self):
        """ length of a curved block edge (two segments) """
        self.mesh.prepare_data()

        self.assertAlmostEqual(
            self.block_0.get_size(0), 1.0397797556255037
        )

    def test_spline_block_size(self):
        """ length of a spline block edge (three or more segments) """
        self.mesh.prepare_data()

        self.assertAlmostEqual(self.block_0.get_size(1), 1.0121046080181824)

    def test_axis_from_pair(self):
        """ return the correct pairs of points along each axis """
        self.mesh.prepare_data()

        pairs = [
            [[0, 1], [3, 2], [4, 5], [7, 6]],
            [[0, 3], [1, 2], [5, 6], [4, 7]],
            [[0, 4], [1, 5], [2, 6], [3, 7]],
        ]

        for i in range(3):
            for j in range(4):
                pair = pairs[i][j]

                v1 = self.block_0.vertices[pair[0]].mesh_index
                v2 = self.block_0.vertices[pair[1]].mesh_index

                axis, direction = self.block_0.get_axis_from_pair([v1, v2])
                self.assertEqual(axis, i)
                self.assertTrue(direction)

                axis, direction = self.block_0.get_axis_from_pair([v2, v1])
                self.assertEqual(axis, i)
                self.assertFalse(direction)

    def test_axis_vertex_pairs(self):
        """ pairs of vertices for wedges """ 
        # create a three-sided pyramid;
        # get_axis_vertex_pairs should return less pairs for triangular faces
        block_points = [
            [0, 0, 0], # 0
            [1, 0, 0], # 1
            [0.5, 1, 0], # 2
            [0.5, 1, 0], # 3

            [0, 0, 1], # 4
            [1, 0, 1], # 5
            [0.5, 1, 1], # 6
            [0.5, 1, 1], # 7
        ]

        block = Block.create_from_points(block_points)
        for i in range(3):
            block.chop(i, count=10)

        mesh = Mesh()
        mesh.add_block(block)
        mesh.prepare_data()

        self.assertEqual(len(block.get_axis_vertex_pairs(0)), 2)
        self.assertEqual(len(block.get_axis_vertex_pairs(1)), 4)
        self.assertEqual(len(block.get_axis_vertex_pairs(2)), 3)

    def test_grading_invert(self):
        """ copy grading when neighbor blocks are upside-down """
        # points & blocks:
        # 3---2---5
        # | 0 | 1 |
        # 0---1---4
        fl = [ # floor
            [0, 0, 0], # 0
            [1, 0, 0], # 1
            [1, 1, 0], # 2
            [0, 1, 0], # 3
            [2, 0, 0], # 4
            [2, 1, 0], # 5
        ]

        # ceiling
        cl = [[p[0], p[1], 1] for p in fl]

        block_0 = Block.create_from_points([
            fl[0], fl[1], fl[2], fl[3],
            cl[0], cl[1], cl[2], cl[3]
        ])
        block_0.chop(0, count=10)
        block_0.chop(1, count=10)
        block_0.chop(2, start_size=0.02, end_size=0.1)
        
        # block_1 is created upside-down
        block_1 = Block.create_from_points([
            cl[2], cl[5], cl[4], cl[1],
            fl[2], fl[5], fl[4], fl[1],
        ])
        block_1.chop(0, count=10)

        self.mesh = Mesh()
        self.mesh.add_block(block_0)
        self.mesh.add_block(block_1)
        self.mesh.prepare_data()

        # also check in real life that calculations are good enough for blockMesh
        self.assertAlmostEqual(
            block_0.grading[2].divisions[0][2],
            1/block_1.grading[2].divisions[0][2])
        #self.run_and_check()

    def test_block_project_face(self):
        self.mesh.prepare_data()

        self.block_0.project_face('bottom', 'terrain')
        self.block_0.project_face('left', 'building')

        expected_list = [
            ['bottom', 'terrain'],
            ['left', 'building']
        ]
        
        self.assertListEqual(
            expected_list, self.block_0.faces
        )

    def test_block_project_face_edges(self):
        # add 4 'project' edges to mesh
        self.block_2.project_face('back', 'terrain', edges=True)
        self.mesh.prepare_data()

        n_project = 0

        for e in self.mesh.edges:
            if e.type == 'project':
                n_project += 1

        self.assertEqual(n_project, 4)

    def test_block_get_face(self):
        self.mesh.prepare_data()

        # blocks 0 and 1 share faces;
        # block_0-right == block_1-left;

        # internal indexes are always between 0...7
        self.assertListEqual(
            list(self.block_0.get_face('right', internal=True)),
            [5, 1, 2, 6] # as defined in Block.face_map
        )
        self.assertListEqual(
            list(self.block_1.get_face('left', internal=True)),
            [4, 0, 3, 7]
        )

        # 'external' faces should be the same
        self.assertListEqual(
            self.block_0.get_face('right'),
            self.block_1.get_face('left')
        )

    def test_block_get_faces(self):
        self.mesh.prepare_data()

        # block_2's patches
        # self.block_2.set_patch(['bottom', 'top', 'left', 'right'], 'walls')
        self.assertListEqual(
            self.block_2.get_faces('walls', internal=True),
            [(0, 1, 2, 3), (4, 5, 6, 7), (4, 0, 3, 7), (5, 1, 2, 6)]
        )

        self.assertListEqual(
            self.block_2.get_faces('walls', internal=False),
            [[2, 9, 12, 13], [6, 11, 14, 15], [6, 2, 13, 15], [11, 9, 12, 14]]
        )

    def test_block_find_edge(self):
        self.mesh.prepare_data()

        # block_0's first edge is curved
        self.assertIsNotNone(self.block_0.find_edge(0, 1))
        self.assertIsNone(self.block_0.find_edge(2, 3))

    def test_block_project_edge(self):
        # this geometry doesn't exist but classy_blocks don't care
        self.block_0.project_edge(2, 3, 'test')

        self.mesh.prepare_data()
        self.assertEqual(self.mesh.edges[-1].type, 'project')
    
    def test_block_project_edge_double(self):
        # projecting an existing edge should not work
        self.block_0.project_edge(0, 1, 'test')
        self.mesh.prepare_data()

        self.assertNotEqual(self.mesh.edges[-1].type, 'project')


class BlockSizingTests(unittest.TestCase):
    def setUp(self):
        points = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],

            [0, 0, 0.5],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ]
        self.block = Block.create_from_points(points)

        for i in range(3):
            self.block.chop(i, count=1)

        self.mesh = Mesh()
        self.mesh.add_block(self.block)
        self.mesh.prepare_data()

    def test_min_block_size(self):
        self.assertAlmostEqual(self.block.get_size(0, take='min'), 1)
        self.assertAlmostEqual(self.block.get_size(1, take='min'), 1)
        self.assertAlmostEqual(self.block.get_size(2, take='min'), 0.5)
        
    def test_max_block_size(self):
        self.assertAlmostEqual(self.block.get_size(0, take='max'), 1.118033989)
        self.assertAlmostEqual(self.block.get_size(1, take='max'), 1.118033989)
        self.assertAlmostEqual(self.block.get_size(2, take='max'), 1)

    def test_avg_block_size(self):
        self.assertAlmostEqual(self.block.get_size(0), 1.0295084971874737)
        self.assertAlmostEqual(self.block.get_size(1), 1.0295084971874737)
        self.assertAlmostEqual(self.block.get_size(2), 0.875)

if __name__ == '__main__':
    unittest.main()