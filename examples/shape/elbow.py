import os
import numpy as np

from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import Elbow

center_point_1 = [0, 0, 0]
radius_point_1 = [1, 0, 0]
normal_1 = [0, 1, 0]

sweep_angle = -np.pi/3
arc_center = [2, 0, 0]
rotation_axis = [0, 0, 1]

radius_2 = 0.4

elbow = Elbow(
    center_point_1, radius_point_1, normal_1,
    sweep_angle, arc_center, rotation_axis, radius_2
)

elbow.set_bottom_patch('inlet')
elbow.set_top_patch('outlet')
elbow.set_outer_patch('walls')

elbow.set_axial_cell_count(25)
elbow.set_radial_cell_count(15)
elbow.set_tangential_cell_count(8)

# TODO: debug (grading is not being copied between blocks)
# TEST
#elbow.set_axial_cell_size(-0.02)
#elbow.set_outer_cell_size(0.02)

mesh = Mesh()
mesh.add_shape(elbow)

mesh.write('case/system/blockMeshDict')
