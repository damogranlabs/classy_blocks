import os

from classy_blocks import Mesh
from classy_blocks.construct.shapes.sphere import EighthSphere


center_point = [0, 0, 0]
radius_point = [0, 0, 1]
normal = [0, 1, 0]

cell_size = 0.1
bl_thickness = 0.01

sphere = EighthSphere(center_point, radius_point, normal)
#sphere.chop_axial(start_size=cell_size)
sphere.chop_radial(start_size=cell_size, end_size=bl_thickness)
sphere.chop_tangential(start_size=cell_size)

#sphere.set_bottom_patch('atmosphere')
#sphere.set_outer_patch('walls') # the same as set_top_patch


mesh = Mesh()
mesh.add(sphere)

mesh.write(os.path.join('..', 'case', 'system', 'blockMeshDict'), debug_path='debug.vtk')
