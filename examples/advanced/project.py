import os

from classy_blocks import Box, Mesh

geometry = {
    'terrain': [
        'type triSurfaceMesh',
        'name terrain',
        'file "terrain.stl"',
    ]
}

mesh = Mesh()

box = Box([-1, -1, -1], [2, 2, 2])

# use edges=True to include all edges;
box.project_side('bottom', 'terrain')

# to project a specific edge only, use block.project_edge()
# extrude.block.project_edge(0, 1, 'terrain')
# extrude.block.project_edge(1, 2, 'terrain')
# extrude.block.project_edge(2, 3, 'terrain')
# extrude.block.project_edge(3, 0, 'terrain')

for axis in (0, 1, 2):
    box.chop(axis, count=20)

box.set_patch('bottom', 'terrain')

mesh.add(box)

#mesh.set_default_patch('atmosphere', 'patch')
#mesh.add_geometry(geometry)

mesh.write(os.path.join('..', 'case', 'system', 'blockMeshDict'), debug_path='debug.vtk')