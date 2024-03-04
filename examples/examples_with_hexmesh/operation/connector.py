import os

import numpy as np

import classy_blocks as cb

box_1 = cb.Box([-1, -1, -1], [1, 1, 1])
box_2 = box_1.copy().rotate(np.pi / 4, [1, 1, 1], [0, 0, 0]).translate([4, 2, 0])

for i in range(3):
    box_1.chop(i, count=10)
    box_2.chop(i, count=10)


connector = cb.Connector(box_1, box_2)
connector.chop(2, count=10)

mesh = cb.Mesh()
mesh.add(box_1)
mesh.add(box_2)
mesh.add(connector)
mesh.set_default_patch("walls", "wall")

# mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")

hexmesh = cb.HexMesh(mesh, quality_metrics=True)
hexmesh.write_vtk(os.path.join("connector.vtk"))
