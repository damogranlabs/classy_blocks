import os

import classy_blocks as cb

mesh = cb.Mesh()

box = cb.Box([0, 0, 0], [1, 1, 1])

# axis 0
box.chop(corner_1=0, corner_2=1, start_size=0.01, end_size=0.1)
box.chop(corner_1=3, corner_2=2, start_size=0.1, end_size=0.05)

# axis 1
box.chop(corner_1=0, corner_2=3, start_size=0.01, end_size=0.05, preserve="start_size")

# axis 2
box.chop(corner_1=0, corner_2=4, count=10)

mesh.add(box)

another = cb.Box([1, 0, 0], [2, 1, 1])
another.points[5].translate([0, 0.5, 0.5])
another.chop(corner_1=0, corner_2=1, start_size=0.2, end_size=0.1)
mesh.add(another)

mesh.set_default_patch("walls", "wall")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"))
