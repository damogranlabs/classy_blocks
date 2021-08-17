import os

from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import Frustum

def get_mesh():
    axis_point_1 = [0, 0, 0]
    axis_point_2 = [2, 2, 0]
    radius_point_1 = [0, 0, 2]
    radius_2 = 0.5

    frustum = Frustum(axis_point_1, axis_point_2, radius_point_1, radius_2)

    frustum.set_bottom_patch('inlet')
    frustum.set_top_patch('outlet')
    frustum.set_outer_patch('walls')

    frustum.set_axial_cell_count(30)
    frustum.set_radial_cell_count(20)
    frustum.set_tangential_cell_count(15)

    frustum.grade_to_size_axial(-0.02)
    frustum.grade_to_size_radial(0.03)

    mesh = Mesh()
    mesh.add(frustum)

    return mesh
