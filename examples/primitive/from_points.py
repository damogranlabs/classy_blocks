from classy_blocks.classes.primitives import Edge
from classy_blocks.classes.block import Block
from classy_blocks.classes.mesh import Mesh

def get_mesh():
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

    block_edges = [
        Edge(0, 1, [0.5, -0.25, 0]), # arc edges
        Edge(4, 5, [0.5, -0.1, 1]),

        Edge(2, 3, [[0.7, 1.3, 0], [0.3, 1.3, 0]]), # spline edges
        Edge(6, 7, [[0.7, 1.1, 1], [0.3, 1.1, 1]])
    ]

    # the most low-level way of creating a block is from 'raw' points
    block = Block.create_from_points(block_points, block_edges)
    block.set_patch(['left', 'right', 'front', 'back'], 'walls')
    block.set_patch('bottom', 'inlet')
    block.set_patch('top', 'outlet')

    block.count_to_size(0, 0.02)
    block.count_to_size(1, 0.05)
    block.count_to_size(2, 0.1)

    mesh = Mesh()
    mesh.add_block(block)

    return mesh