import os

from classy_blocks import Box, Mesh

box = Box([-1, -2, -4], [4, 2, 1])

# direction of corners 0-1
box.chop(0, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=False)
box.chop(0, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=True)

# direction of corners 1-2
box.chop(1, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=False)
box.chop(1, start_size=0.02, c2c_expansion=1.2, length_ratio=0.5, invert=True)

# extrude direction
box.chop(2, c2c_expansion=1, count=20)

mesh = Mesh()
mesh.add(box)
mesh.set_default_patch('walls', 'wall')

mesh.write(os.path.join('..', 'case', 'system', 'blockMeshDict'))
