import os

import classy_blocks as cb
from classy_blocks.construct.flat.sketches.disk import QuarterDisk
from classy_blocks.util import functions as f

mesh = cb.Mesh()

axis_point_1 = f.vector(0.0, 0.0, 0.0)
axis_point_2 = f.vector(5.0, 5.0, 0.0)
radius_point_1 = f.vector(0.0, 0.0, 2.0)

# make a disk with small core block - yields particularly bad cell sizes with normal chops
QuarterDisk.core_ratio = 0.4

quarter_disk = QuarterDisk(axis_point_1, radius_point_1, axis_point_1 - axis_point_2)
quarter_cylinder = cb.ExtrudedShape(quarter_disk, f.norm(axis_point_2 - axis_point_1))

mesh.add(quarter_cylinder)

# Use an automatic grader that will try to make cells in neighbouring blocks
# as similar in size as possible
grader = cb.SmoothGrader(mesh, 0.05)
grader.grade()

mesh.set_default_patch("walls", "wall")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
