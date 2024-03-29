import os

import classy_blocks as cb

base = cb.Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [cb.Arc([0.5, -0.2, 0]), None, None, None])

extrude = cb.Extrude(base, [0.5, 0.5, 3])

# direction of corners 0-1
extrude.chop(0, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=False)
extrude.chop(0, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=True)

# direction of corners 1-2
extrude.chop(1, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=False)
extrude.chop(1, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=True)

# extrude direction
extrude.chop(2, c2c_expansion=1, count=20)

mesh = cb.Mesh()
mesh.add(extrude)
mesh.set_default_patch("walls", "wall")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"))
