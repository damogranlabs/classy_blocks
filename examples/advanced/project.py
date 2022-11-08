from classy_blocks import Face, Extrude, Mesh

geometry = {
    'terrain': [
        'type triSurfaceMesh',
        'name terrain',
        'file "terrain.stl"',
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

    # use edges=True to include all edges;
    extrude.block.project_face('bottom', 'terrain', edges=True)

    # to project a specific edge only, use block.project_edge()
    # extrude.block.project_edge(0, 1, 'terrain')
    # extrude.block.project_edge(1, 2, 'terrain')
    # extrude.block.project_edge(2, 3, 'terrain')
    # extrude.block.project_edge(3, 0, 'terrain')

    for axis in range(3):
        extrude.chop(axis, count=20)

    extrude.set_patch('bottom', 'terrain')

    mesh = Mesh()
    mesh.add(extrude)
    mesh.set_default_patch('atmosphere', 'patch')
    mesh.add_geometry(geometry)

    return mesh