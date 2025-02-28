import os

import classy_blocks as cb

box = cb.Box([-1, -2, -4], [4, 2, 1])

# direction of corners 0-1
box.chop(0, length_ratio=0.5, start_size=0.02, c2c_expansion=1.2)
box.chop(0, length_ratio=0.5, end_size=0.02, c2c_expansion=1 / 1.2)

# direction of corners 1-2
box.chop(1, length_ratio=0.5, start_size=0.02, c2c_expansion=1.2)
box.chop(1, length_ratio=0.5, end_size=0.02, c2c_expansion=1 / 1.2)

# extrude direction
box.chop(2, c2c_expansion=1, count=20)

mesh = cb.Mesh()
mesh.add(box)
mesh.set_default_patch("walls", "wall")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"))
