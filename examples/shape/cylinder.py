import os

from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.shapes import Cylinder

axis_point_1 = [0, 0, 0]
axis_point_2 = [5, 5, 0]
radius_point_1 = [0, 0, 2]

cylinder = Cylinder(axis_point_1, axis_point_2, radius_point_1)

cylinder.set_bottom_patch('inlet')
cylinder.set_top_patch('outlet')
cylinder.set_outer_patch('walls')

cylinder.set_axial_cell_count(30)
cylinder.set_radial_cell_count(20)
cylinder.set_tangential_cell_count(15)

cylinder.set_axial_cell_size(-0.05)
cylinder.set_outer_cell_size(0.03)

mesh = Mesh()
mesh.add_shape(cylinder)

mesh.write('case/system/blockMeshDict')
