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

# cylinder.chop_axial(count=30)
# cylinder.chop_radial(start_size=core_size, end_size=bl_thickness)
# cylinder.chop_tangential(start_size=core_size)

# chop a single edge of a single operation
# case 1: remove grading
# cylinder.shell[0].chop_edge(0, 1, c2c_expansion=1)
# case 2: make a thicker first cell
# cylinder.shell[1].chop_edge(0, 1, end_size=5 * bl_thickness)
# case 3: weird random multigrading
# cylinder.shell[2].chop_edge(0, 1, length_ratio=0.5, count=2)
# cylinder.shell[2].chop_edge(0, 1, c2c_expansion=1 / 1.5)
# case 4: chop one block differently so that neighbour block will be edge-graded
# cylinder.shell[1].chop(2, count=30, start_size=0.01)


mesh.add(cylinder)
cylinder.set_start_patch("walls")
cylinder.set_end_patch("walls")
mesh.modify_patch("walls", "wall")

grader = cb.InflationGrader(mesh, bl_thickness, core_size)
grader.grade()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
