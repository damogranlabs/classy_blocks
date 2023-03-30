import os
import classy_blocks as cb

mesh = cb.Mesh()

axis_point_1 = [0., 0., 0.]
axis_point_2 = [2., 2., 0.]
radius_point_1 = [0., 0., 2.]
radius_2 = 0.5

bl_thickness = 0.01
core_size = 0.1

# A note about radius_mid;
# it can be used to create shapes of revolution with curved sides;
# however, due to the way blockMesh face creation works, the result won't
# be totally 'round'.
# Also, in drastic cases, non-orthogonality at beginning/end face will be high
# because of sharp edges; in those cases it is better to use RevolvedRing combined with
# Cylinder/Frustum with non-flat start/end faces.
frustum = cb.Frustum(axis_point_1, axis_point_2, radius_point_1, radius_2, radius_mid=1.1)

frustum.set_start_patch('inlet')
frustum.set_outer_patch('walls')
frustum.set_end_patch('outlet')

frustum.chop_axial(count=30)
frustum.chop_radial(start_size=core_size, end_size=bl_thickness)
frustum.chop_tangential(start_size=core_size)

mesh.add(frustum)
mesh.write(os.path.join('..', 'case', 'system', 'blockMeshDict'), debug_path='debug.vtk')
