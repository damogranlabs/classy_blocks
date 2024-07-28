import os

import classy_blocks as cb
from classy_blocks.construct.assemblies.joints import LJoint
from classy_blocks.util import functions as f

mesh = cb.Mesh()

axis_point_1 = [0.0, 0.0, 0.0]
axis_point_2 = [5.0, 5.0, 0.0]
radius_point_1 = [0.0, 0.0, 2.0]

joint = LJoint(axis_point_1, axis_point_2, radius_point_1)

cell_size = f.norm(radius_point_1) / 10

joint.chop_axial(start_size=cell_size * 5, end_size=cell_size)
joint.chop_radial(start_size=cell_size, end_size=cell_size / 10)
joint.chop_tangential(start_size=cell_size)

joint.set_outer_patch("walls")
joint.set_hole_patch(0, "inlet")
joint.set_hole_patch(1, "outlet")

mesh.add(joint)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
