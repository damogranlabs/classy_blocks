import os

import classy_blocks as cb

mesh = cb.Mesh()

axis_point_1 = [0.0, 0.0, 0.0]
axis_point_2 = [2.0, 0.0, 0.0]
radius_point_1 = [0.0, 0.0, 2.0]
radius_2 = 0.5

bl_thickness = 0.01
core_size = 0.1

frustum = cb.Frustum(axis_point_1, axis_point_2, radius_point_1, radius_2, radius_mid=1.1)
frustum.set_end_patch("inlet")
frustum.chop_axial(count=30)
frustum.chop_radial(start_size=core_size, end_size=bl_thickness)
frustum.chop_tangential(start_size=core_size)
mesh.add(frustum)

# mirroring an operation or a shape will
# not invert blocks' orientation;
# their 'start' and 'end' patches (axis 2) will
# point in the same direction as the original blocks/operations
mirror = frustum.copy().mirror([1, 0, 0])
mirror.set_start_patch("outlet")
mirror.chop_axial(count=30)
mesh.add(mirror)

mesh.set_default_patch("walls", "wall")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
