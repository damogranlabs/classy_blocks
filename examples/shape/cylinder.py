import os

import classy_blocks as cb
from classy_blocks.construct.flat.sketches.disk import QuarterDisk

# QuarterDisk.diagonal_ratio = 0.8 Is defined from side_ratio
QuarterDisk.side_ratio = 0.8  # Default is 0.8

mesh = cb.Mesh()

axis_point_1 = [0.0, 0.0, 0.0]
axis_point_2 = [5.0, 5.0, 0.0]
radius_point_1 = [0.0, 0.0, 2.0]

cylinder = cb.Cylinder(axis_point_1, axis_point_2, radius_point_1)

cylinder.set_start_patch("inlet")
cylinder.set_end_patch("outlet")
cylinder.set_outer_patch("walls")

bl_thickness = 0.05
core_size = 0.2

cylinder.chop_axial(count=30)
cylinder.chop_radial(start_size=core_size, end_size=bl_thickness)
cylinder.chop_tangential(start_size=core_size)

mesh.add(cylinder)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
