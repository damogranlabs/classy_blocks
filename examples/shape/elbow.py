import os
import numpy as np

from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import Elbow

radius_1 = 1
center_point_1 = [0, 0, 0]
radius_point_1 = [radius_1, 0, 0]
normal_1 = [0, 1, 0]

sweep_angle = -np.pi/3
arc_center = [2, 0, 0]
rotation_axis = [0, 0, 1]

radius_2 = 0.4
cell_size = 0.05

elbow = Elbow(
    center_point_1, radius_point_1, normal_1,
    sweep_angle, arc_center, rotation_axis, radius_2
)

elbow.set_bottom_patch('inlet')
elbow.set_top_patch('outlet')
elbow.set_outer_patch('walls')

# set cell sizes
elbow.count_to_size_axial(cell_size)
elbow.count_to_size_radial(cell_size)
elbow.count_to_size_tangential(cell_size)

# or counts
#elbow.set_axial_cell_count(25)
#elbow.set_radial_cell_count(15)
#elbow.set_tangential_cell_count(8)

# grading in axial direction;
# negative to set cell size on the opposite side
elbow.grade_to_size_axial(-cell_size * radius_2/radius_1)

mesh = Mesh()
mesh.add(elbow)

mesh.write('case/system/blockMeshDict')
