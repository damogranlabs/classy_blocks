from classy_blocks.data import edges
from classy_blocks.items.block import Block
from classy_blocks.mesh import Mesh

mesh = Mesh()

# the most low-level way of creating a block is from 'raw' points
block_0_points = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],

    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
]

block_0 = Block(block_0_points)

# add edges
block_0.add_edge(0, 1, edges.Arc([0.5, -0.25, 0])) # arc edge
block_0.add_edge(4, 5, edges.Arc([0.5, -0.1, 1]))
block_0.add_edge(2, 3, edges.Spline([[0.7, 1.3, 0], [0.3, 1.3, 0]])) # spline edge
block_0.add_edge(6, 7, edges.PolyLine([[0.7, 1.1, 1], [0.3, 1.1, 1]])) # weird edge
block_0.add_edge(0, 4, edges.Origin([0.5, 0.5, 0.5])) # ESI-CFD's alternative definition
block_0.add_edge(1, 5, edges.Angle(3.14159/2, [0, 1, 0])) # Foundation's alternative definition

block_0.set_patch(['left', 'right', 'front', 'back'], 'walls', 'wall')
block_0.set_patch('bottom', 'inlet')

#block.project_face('bottom', 'terrain', edges=True)

block_0.chop(0, start_size=0.02, c2c_expansion=1.1)
block_0.chop(1, start_size=0.01, c2c_expansion=1.2)
block_0.chop(2, start_size=0.1, c2c_expansion=1)

# TODO: replace with mesh.add()
mesh.add_block(block_0)

# another block!
block_1_points = block_0_points[4:] + [
    [0, 0, 1.7],
    [1, 0, 1.8],
    [1, 1, 1.9],
    [0, 1, 2],
]
block_1 = Block(block_1_points)
block_1.set_patch(['left', 'right', 'front', 'back'], 'walls', 'wall')
block_1.set_patch('top', 'outlet')

block_1.chop(2, length_ratio=0.5, start_size=0.02, c2c_expansion=1.2, invert=False)
block_1.chop(2, length_ratio=0.5, start_size=0.02, c2c_expansion=1.2, invert=True)

# TODO: replace with mesh.add()
mesh.add_block(block_1)


mesh.write('../case/system/blockMeshDict')
