import os
import classy_blocks as cb

center = [0.0, 0.0, 0.0]
radius_point = [0.0, 0.0, 1.0]
normal = [0.0, 1.0, 0.0]

cell_size = 0.1
bl_thickness = 0.01

sphere = cb.Hemisphere(center, radius_point, normal)
sphere.chop_axial(start_size=cell_size)
sphere.chop_radial(start_size=cell_size, end_size=bl_thickness)
sphere.chop_tangential(start_size=cell_size)

sphere.set_start_patch('atmosphere') # the 'flat' part of the hemisphere
sphere.set_outer_patch('walls')

mesh = cb.Mesh()
mesh.add(sphere)

mesh.modify_patch('walls', 'wall')
mesh.write(os.path.join('..', 'case', 'system', 'blockMeshDict'), debug_path='debug.vtk')
