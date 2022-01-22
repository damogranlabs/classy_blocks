import os

from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import Frustum

def get_mesh():
    axis_point_1 = [0, 0, 0]
    axis_point_2 = [2, 2, 0]
    radius_point_1 = [0, 0, 2]
    radius_2 = 0.5

    bl_thickness = 0.01
    core_size = 0.1

    frustum = Frustum(axis_point_1, axis_point_2, radius_point_1, radius_2)

    frustum.set_bottom_patch('inlet')
    frustum.set_top_patch('outlet')
    frustum.set_outer_patch('walls')

    frustum.chop_axial(count=30)
    frustum.chop_radial(start_size=bl_thickness, end_size=core_size)
    frustum.chop_tangential(start_size=core_size)

    mesh = Mesh()
    mesh.add(frustum)

    return mesh
