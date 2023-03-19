import os

from classy_blocks import Box, Mesh

geometry = {
    'terrain': [
        'type triSurfaceMesh',
        'name terrain',
        'file "terrain.stl"',
    ],
    'left_wall': [
        'type       searchablePlane',
        'planeType  pointAndNormal',
        'point      (-1 0 0)',
        'normal     (1  0  0)',
    ],
    'front_wall': [
        'type       searchablePlane',
        'planeType  pointAndNormal',
        'point      (0 -1 0)',
        'normal     (0  1  0)',
    ]
}

mesh = Mesh()

# 'miss' the vertices deliberately;
# project them to geometry later
box = Box([-0.9, -0.9, -0.9], [1., 1., 1.])

# project misplaced vertices
box.project_corner(0, ['terrain', 'left_wall', 'front_wall'])
box.project_corner(1, 'terrain')
box.project_corner(2, 'terrain')

# project a face to geometry;
# when using Loft/Extrude/Revolve, you could specify
# those when creating a Face; you'd still have to 
# project other sides this way
box.project_side('bottom', 'terrain')

# projection of an edge to a surface will move it in various directions,
# depending on the geometry, distorting the box's side
box.project_edge(0, 1, 'terrain')

# to avoid that, project an edge to two surfaces;
# this will make it stick to their intersection
box.project_edge(3, 0, ['terrain', 'left_wall'])

# an edge can remain 'not projected' but this can cause
# bad quality cells if geometries differ enough
# extrude.block.project_edge(1, 2, 'terrain')
# extrude.block.project_edge(2, 3, 'terrain')
# extrude.block.project_edge(3, 0, 'terrain')

for axis in (0, 1, 2):
    box.chop(axis, count=20)

box.set_patch('bottom', 'terrain')

mesh.add(box)

mesh.set_default_patch('atmosphere', 'patch')
mesh.add_geometry(geometry)

mesh.write(os.path.join('..', 'case', 'system', 'blockMeshDict'), debug_path='debug.vtk')
