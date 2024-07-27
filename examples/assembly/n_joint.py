import os

import classy_blocks as cb
from classy_blocks.construct.assemblies.joints import NJoint

mesh = cb.Mesh()

axis_point_1 = [0.0, 0.0, 0.0]
axis_point_2 = [5.0, 5.0, 0.0]
radius_point_1 = [0.0, 0.0, 2.0]

# cylinder = cb.Cylinder(axis_point_1, axis_point_2, radius_point_1)

joint = NJoint([0, -1, 0], [0, 0, 0], [0.2, -1, 0], 3)

for operation in joint.operations:
    for i in range(3):
        operation.chop(i, count=10)

mesh.add(joint)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
