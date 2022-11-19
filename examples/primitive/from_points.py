from classy_blocks import Block, Mesh

def get_mesh():
    mesh = Mesh()

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

    # the most low-level way of creating a block is from 'raw' points
    block = Block.create_from_points(block_points)

    # add edges
    block.add_edge(0, 1, [0.5, -0.25, 0]) # arc edge
    block.add_edge(4, 5, [0.5, -0.1, 1])
    block.add_edge(2, 3, [[0.7, 1.3, 0], [0.3, 1.3, 0]]) # spline edge
    block.add_edge(6, 7, [[0.7, 1.1, 1], [0.3, 1.1, 1]], kind='polyLine') # weird edge

    block.set_patch(['left', 'right', 'front', 'back'], 'walls')
    block.set_patch('bottom', 'inlet')

    block.chop(0, start_size=0.02, c2c_expansion=1.1)
    block.chop(1, start_size=0.01, c2c_expansion=1.2)
    block.chop(2, start_size=0.1, c2c_expansion=1)

    mesh.add(block)

    # another block!
    block_points = block_points[4:] + [
        [0, 0, 1.7],
        [1, 0, 1.8],
        [1, 1, 1.9],
        [0, 1, 2],
    ]
    block = Block.create_from_points(block_points)
    block.set_patch(['left', 'right', 'front', 'back'], 'walls')
    block.set_patch('top', 'outlet')

    block.chop(2, length_ratio=0.5, start_size=0.02, c2c_expansion=1.2, invert=False)
    block.chop(2, length_ratio=0.5, start_size=0.02, c2c_expansion=1.2, invert=True)

    mesh.add(block)

    return mesh
