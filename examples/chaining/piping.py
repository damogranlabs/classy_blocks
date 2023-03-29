import os

import numpy as np

import classy_blocks as cb

mesh = cb.Mesh()

cell_size = 0.08


elbow = cb.Elbow(
    [0, 0, 0], # center_point_1
    [1, 0, 0], # radius_point_1
    [0, 1, 0], # normal_1
    -np.pi/2, # sweep_angle
    [2, 0, 0], # arc_center
    [0, 0, 1], # rotation_axis
    0.5 # radius_2
)

chained = cb.Elbow.chain(
    elbow, # source
    -np.pi/2, # sweep_angle
    [2, 0, 0], # arc_center
    [0, 0, 1], # rotation_axis
    1, # radius_2
)

elbow.chop_axial(start_size=cell_size)
elbow.chop_radial(start_size=cell_size)
elbow.chop_tangential(start_size=cell_size)

chained.chop_axial(start_size=cell_size)

mesh.add(elbow)
mesh.add(chained)
mesh.write(os.path.join('..', 'case', 'system', 'blockMeshDict'), debug_path='debug.vtk')
