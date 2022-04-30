import os
import numpy as np

from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import Elbow

def get_mesh():
    mesh = Mesh()

    radius_1 = 1
    center_point_1 = [0, 0, 0]
    radius_point_1 = [radius_1, 0, 0]
    normal_1 = [0, 1, 0]

    sweep_angle = -np.pi/3
    arc_center = [2, 0, 0]
    rotation_axis = [0, 0, 1]

    radius_2 = 0.4
    boundary_size = 0.01
    core_size = 0.08

    elbow = Elbow(
        center_point_1, radius_point_1, normal_1,
        sweep_angle, arc_center, rotation_axis, radius_2
    )

    elbow.set_bottom_patch('inlet')
    elbow.set_outer_patch('walls')

    # counts and gradings
    elbow.chop_tangential(start_size=core_size)
    elbow.chop_radial(start_size=core_size, end_size=boundary_size)
    elbow.chop_axial(start_size=2*core_size)

    mesh.add(elbow)

    return mesh
