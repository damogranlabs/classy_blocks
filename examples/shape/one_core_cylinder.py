import os

import classy_blocks as cb
from classy_blocks.construct.flat.sketches.disk import OneCoreDisk
from classy_blocks.util import functions as f

# A simpler version of a Cylinder with only a single block in the middle;
# easier to block but slightly worse cell quality;
# currently there is no OneCoreCylinder shape so it has to be extruded;
# two lines are needed instead of one.

mesh = cb.Mesh()

axis_point_1 = f.vector(0.0, 0.0, 0.0)
axis_point_2 = f.vector(5.0, 5.0, 0.0)
radius_point_1 = f.vector(0.0, 0.0, 2.0)


one_core_disk = OneCoreDisk(axis_point_1, radius_point_1, axis_point_1 - axis_point_2)
quarter_cylinder = cb.ExtrudedShape(one_core_disk, f.norm(axis_point_2 - axis_point_1))
quarter_cylinder.chop(0, count=5)
quarter_cylinder.chop(1, count=10)
quarter_cylinder.chop(2, count=20)
mesh.add(quarter_cylinder)

mesh.set_default_patch("walls", "wall")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
