"""A simple cylinder but with custom grading of selected edges"""

import os

import classy_blocks as cb

mesh = cb.Mesh()

axis_point_1 = [0, 0, 0]
axis_point_2 = [1, 0, 0]
radius_point_1 = [0, 1, 0]

cylinder = cb.Cylinder(axis_point_1, axis_point_2, radius_point_1)

cylinder.set_start_patch("inlet")
cylinder.set_end_patch("outlet")
cylinder.set_outer_patch("walls")

# Chop and grade
bl_thickness = 1e-3
core_size = 0.1

# Edge chopping can only be done on a completely specified mesh;
cylinder.chop_axial(count=30)
cylinder.chop_radial(start_size=core_size, end_size=bl_thickness)
cylinder.chop_tangential(start_size=core_size)

# After all edges have been specified (chopped),
# specific ones can be changed manually.
# Keep in mind that count is already fixed and cannot be changed.

# Case 1: remove grading
cylinder.shell[0].chop_edge(0, 1, c2c_expansion=1)
# Case 2: make a thicker first cell
cylinder.shell[1].chop_edge(0, 1, end_size=5 * bl_thickness)
# Case 3: weird random multigrading
cylinder.shell[2].chop_edge(0, 1, length_ratio=0.5, count=2)
cylinder.shell[2].chop_edge(0, 1, c2c_expansion=1 / 1.5)
# Case 4: chop one block differently so that neighbour block will be edge-graded
cylinder.shell[1].chop(2, count=30, start_size=0.01)


mesh.add(cylinder)
cylinder.set_start_patch("walls")
cylinder.set_end_patch("walls")
mesh.modify_patch("walls", "wall")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
