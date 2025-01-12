import os

import classy_blocks as cb
from classy_blocks.construct.flat.sketches.disk import DiskBase

DiskBase.core_ratio = 0.4  # Default is 0.8

mesh = cb.Mesh()

axis_point_1 = [0.0, 0.0, 0.0]
axis_point_2 = [5.0, 5.0, 0.0]
radius_point_1 = [0.0, 0.0, 2.0]

cylinder = cb.Cylinder(axis_point_1, axis_point_2, radius_point_1)

cylinder.set_start_patch("inlet")
cylinder.set_end_patch("outlet")
cylinder.set_outer_patch("walls")

# if curved core edges get in the way (when moving vertices, optimization, ...),
# remove them with this method:
cylinder.remove_inner_edges(start=False, end=True)

bl_thickness = 1e-3
core_size = 0.1

# manual grading
# cylinder.chop_axial(count=30)
# cylinder.chop_radial(start_size=core_size, end_size=bl_thickness)
# cylinder.chop_tangential(start_size=core_size)

mesh.add(cylinder)
mesh.modify_patch("walls", "wall")

# automatic grading
# grader = cb.SmoothGrader(mesh, 0.1)
grader = cb.InflationGrader(mesh, bl_thickness, core_size)
grader.grade()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
