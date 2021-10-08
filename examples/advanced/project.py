import os

from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.operations import Face, Extrude, Loft

geometry = {
    'terrain': [
        'type triSurfaceMesh;',
        'name terrain;',
        'file "terrain.stl";',
    ]
}

def get_mesh():
    base = Face([
            [-1, -1, -1],
            [1, -1,  -1],
            [1,  1,  -1],
            [-1, 1,  -1]
    ])

    extrude = Extrude(base, [0, 0, 2])

    extrude.block.project_face('bottom', 'terrain')

    extrude.block.project_edge(0, 1, 'terrain')
    extrude.block.project_edge(1, 2, 'terrain')
    extrude.block.project_edge(2, 3, 'terrain')
    extrude.block.project_edge(3, 0, 'terrain')

    extrude.set_cell_count(0, 20)
    extrude.set_cell_count(1, 20)
    extrude.set_cell_count(2, 20)

    extrude.set_patch('bottom', 'terrain')

    mesh = Mesh()
    mesh.add(extrude)
    mesh.set_default_patch('atmosphere', 'patch')

    return mesh