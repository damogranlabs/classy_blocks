from classy_blocks.mesh import Mesh
from classy_blocks.data.block import BlockData

mesh = Mesh()

# the most low-level way of creating a block is from 'raw' points
block_points = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],

    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
]

block = BlockData(block_points)

# add edges
block.add_edge(0, 1, 'arc', [0.5, -0.25, 0]) # arc edge
block.add_edge(4, 5, 'arc', [0.5, -0.1, 1])
block.add_edge(2, 3, 'spline', [[0.7, 1.3, 0], [0.3, 1.3, 0]]) # spline edge
block.add_edge(6, 7, 'polyLine', [[0.7, 1.1, 1], [0.3, 1.1, 1]]) # weird edge

block.set_patch(['left', 'right', 'front', 'back'], 'walls', 'wall')
block.set_patch('bottom', 'inlet')

#block.project_face('bottom', 'terrain', edges=True)

#block.chop(0, start_size=0.02, c2c_expansion=1.1)
#block.chop(1, start_size=0.01, c2c_expansion=1.2)
#block.chop(2, start_size=0.1, c2c_expansion=1)

mesh.add(block)

# another block!
block_points = block_points[4:] + [
    [0, 0, 1.7],
    [1, 0, 1.8],
    [1, 1, 1.9],
    [0, 1, 2],
]
block = BlockData(block_points)
block.set_patch(['left', 'right', 'front', 'back'], 'walls', 'wall')
block.set_patch('top', 'outlet')

# block.chop(2, length_ratio=0.5, start_size=0.02, c2c_expansion=1.2, invert=False)
# block.chop(2, length_ratio=0.5, start_size=0.02, c2c_expansion=1.2, invert=True)

mesh.add(block)


mesh.write('../case/system/blockMeshDict')