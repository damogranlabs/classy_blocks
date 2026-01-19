import os

import classy_blocks as cb
from classy_blocks.construct.flat.sketches.disk import WrappedDisk

# A simpler version of a Cylinder with only a single block in the middle;
# easier to block but slightly worse cell quality;
# currently there is no OneCoreCylinder shape so it has to be extruded;
# two lines are needed instead of one.

mesh = cb.Mesh()

center_1 = (0.0, 0.0, 0.0)
corner_point = (-2, -2, 0)
radius = 1
height = 2

cell_size = 0.1

one_core_disk = WrappedDisk(center_1, corner_point, radius, [0, 0, 1])
cylinder = cb.ExtrudedShape(one_core_disk, height)
cylinder.chop(0, start_size=cell_size)
cylinder.chop(1, start_size=cell_size)
cylinder.chop(2, start_size=cell_size)
mesh.add(cylinder)

mesh.set_default_patch("walls", "wall")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
