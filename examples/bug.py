import numpy as np

import classy_blocks as cb

box = cb.Box([0, 0, 0], [1, 1, 1])

base_face = box.bottom_face
neighbour_face = base_face.copy().translate([4, 0, 0])
common_face = base_face.copy().rotate(np.pi / 2, [0, 1, 0]).translate([2, 0, 2])

left_loft = cb.Loft(base_face, common_face)
right_loft = cb.Loft(neighbour_face, common_face.copy().shift(2).invert())

for axis in range(3):
    left_loft.chop(axis, start_size=0.05, total_expansion=5)

right_loft.chop(2, count=10)

mesh = cb.Mesh()
mesh.add(left_loft)
mesh.add(right_loft)

mesh.write("case/system/blockMeshDict", debug_path="debug.vtk")
