import os

import classy_blocks as cb

mesh = cb.Mesh()

axis_point_1 = [0, 0, 0]
axis_point_2 = [0, 0, 1]
radius_point_1 = [1, 0, 0]

cylinder = cb.Cylinder(axis_point_1, axis_point_2, radius_point_1)

cylinder.set_start_patch("inlet")
cylinder.set_end_patch("outlet")
cylinder.set_outer_patch("walls")

# If curved core edges get in the way (when moving vertices, optimization, ...),
# remove them with this method:
cylinder.remove_inner_edges(start=False, end=True)

# Chop and grade
bl_thickness = 1e-3
core_size = 0.1

cylinder.chop_axial(count=30)
cylinder.chop_radial(start_size=core_size, end_size=bl_thickness)
cylinder.chop_tangential(start_size=core_size)

mesh.add(cylinder)
mesh.modify_patch("walls", "wall")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
