import numpy as np

from classy_blocks.process.mesh import Mesh
from classy_blocks.construct.walls import ElbowWall

def get_mesh():
    mesh = Mesh()


    radius_1 = 1
    center_point_1 = [0, 0, 0]
    radius_point_1 = [radius_1, 0, 0]
    normal_1 = [0, 1, 0]
    thickness_1 = 0.1

    sweep_angle = -np.pi/3
    arc_center = [2, 0, 0]
    rotation_axis = [0, 0, 1]

    radius_2 = 0.4
    thickness_2 = 0.2

    elbow = ElbowWall(
        center_point_1, radius_point_1, normal_1, thickness_1,
        sweep_angle, arc_center, rotation_axis,
        radius_2, thickness_2
    )

    elbow.chop_axial(start_size=0.05)
    elbow.chop_tangential(start_size=0.1)

    # chop twice to get grading on both sides of 'the wall'
    bl_thickness = 0.005
    c2c_expansion = 1.2
    elbow.chop_radial(length_ratio=0.5, start_size=bl_thickness, c2c_expansion=c2c_expansion)
    elbow.chop_radial(length_ratio=0.5, end_size=bl_thickness, c2c_expansion=1/c2c_expansion)

    elbow.set_bottom_patch('inlet')
    elbow.set_top_patch('outlet')

    elbow.set_inner_patch('inner_wall')
    elbow.set_outer_patch('outer_wall')


    mesh.add(elbow)

    return mesh
