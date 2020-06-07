import os

from classes.primitives import Edge
from classes.mesh import Mesh

from operations.base import Face
from operations.operations import Loft

def create():
    bottom_face = Face(
        [ # vertices
            [0, 0, 0], # 0
            [1, 0, 0], # 1
            [1, 1, 0], # 2
            [0, 1, 0]  # 3
        ],
        [
            [0.5, -0.25, 0], # edge between 0 and 1
            None,            # 1 and 2
            [0.5, 1.25, 0],  # 2 and 3
            None             # 3 and 0
        ]
    )

    top_face = Face(
        [ #
            [0, 0, 2], # 4
            [1, 0, 2], # 5
            [1, 1, 2], # 6
            [0, 1, 2]  # 7
        ],
        [
            None,
            [1.25, 0.5, 2],
            None,
            [-0.25, 0.5, 2]
        ]
    )

    side_edges = [
        [ [0.02, 0.02, 0.5], [-0.02, -0.02, 1.0], [0.02, 0.02, 1.5], ], # 0-4 (spline edge)
        [0.9, 0.1, 1], # 1-5
        [0.9, 0.9, 1], # 2-6
        [0.1, 0.9, 1]  # 3-7
    ]

    loft = Loft(bottom_face, top_face, side_edges)
    loft.set_cell_count(2, 30)
    loft.set_cell_size(2, 0.01)

    mesh = Mesh()
    mesh.add_block(loft.block)
    mesh.write('util/blockMeshDict.template', 'examples/meshCase/system/blockMeshDict')

    # run blockMesh
    os.system("blockMesh -case examples/meshCase")
