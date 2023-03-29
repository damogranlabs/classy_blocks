import unittest

import numpy as np

from classy_blocks.construct.shapes.elbow import Elbow
from classy_blocks.construct.shapes.cylinder import Cylinder
from classy_blocks.mesh import Mesh

class ElbowChainingTests(unittest.TestCase):
    """Chaining of elbow to everything elbow-chainable"""
    def setUp(self):
        self.elbow = Elbow(
            [0, 0, 0], # center_point_1
            [1, 0, 0], # radius_point_1
            [0, 1, 0], # normal_1
            -np.pi/2, # sweep angle
            [2, 0, 0], # arc_center
            [0, 0, 1], # rotation_axis
            1.0 # radius_2
        )

        self.mesh = Mesh()
    
    def check_success(self, chained_shape):
        """adds the chained stuff to mesh and
        checks the number of vertices as a measurement of success"""
        self.mesh.add(self.elbow)
        self.mesh.add(chained_shape)

        self.mesh.assemble()

        self.assertEqual(len(self.mesh.block_list.blocks), 24)
        self.assertEqual(len(self.mesh.vertex_list.vertices), 3*17)

    def test_to_elbow_end(self):
        """Chain an elbow to an elbow on an end sketch"""
        chained = Elbow.chain(
            self.elbow, # source
            -np.pi/2, # sweep_angle
            [2, 0, 0], # arc_center
            [0, 0, 1], # rotation_axis
            1, # radius_2
            False) # start_face
        
        self.check_success(chained)
    
    def test_to_elbow_start(self):
        """Chain an elbow to an elbow on a start sketch"""
        _ = Elbow.chain(self.elbow, -np.pi, [-2, 0, 0], [-1, 0, 0], 1, True)
    
    def test_to_cylinder_start(self):
        """Chain an elbow to a cylinder on an end sketch"""
        _ = Cylinder.chain(self.elbow, 1)
    