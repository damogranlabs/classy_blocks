#!/usr/bin/python3
import os

import sys
sys.path.append(os.path.abspath("."))

import classy_blocks
from functions import vector, rotate, norm, deg2rad
import numpy as np


# create a 3d mesh of a pipe with rapid diameter increase (short cone):
# nozzle         | cone |             outside           |        
#                |      _________________________________
#                |     /                              o |
#                |    /                               u |
#                |   /                                t |
#                |  /                                 l |
#                  /                                  e |
# ________________/                                   t |
# inlet                                                 |
# ____  __  _____  __  _____  __  _____  __  _____  __  _____  axis of rotation (not symmetry/wedge!)


# geometry data:
# wedge angle
alpha = deg2rad(5)

d_nozzle = 0.8e-3
l_nozzle = 5.0e-3
l_cone = 1.0e-3

d_pipe = 4.0e-3
l_pipe = 66e-3

# number of block cells:
n_cells = 5

# divide number of cells outside important areas
# and apply grading to match sizes; to save on cell count
cell_dilution = 3

### create a classy mesh
mesh = classy_blocks.Mesh()

# a cylinder with given parameters (along x-axis)
def cylinder(x_1, x_2, r_1, r_2=None):
    if r_2 is None:
        r_2 = r_1

    # all points are defined in XY-plane, then rotated by a specified angle
    # around x-axis;
    e_angles = np.linspace(0, 2*np.pi, 4, endpoint=False) # edge angles
    p_angles = e_angles + np.pi/4 # point angles

    # start and end face of cylinder and their points
    def face(x, r):
        pi = vector(x, r/2, 0) # inner block
        po = vector(x, r, 0) # outer blocks
        e = vector(x, r, 0) # starting point edges

        # block and edge points
        points = [rotate(pi, -a) for a in p_angles] + \
            [rotate(po, -a) for a in p_angles]

        edges = [rotate(e, -a) for a in e_angles]

        return points, edges

    p1, e1 = face(x_1, r_1)
    p2, e2 = face(x_2, r_2)

    # number of cells
    dl = x_2 - x_1
    cell_size = r_1/n_cells
    n_cells_axial = int(dl/cell_size)

    # blocks, edges, patches
    def block(indices, edge_index=None):
        points = list(np.take(p1, indices, axis=0)) + \
            list(np.take(p2, indices, axis=0))

        vertices = mesh.add_vertices(points)

        if edge_index is not None:
            mesh.add_edge(vertices[1], vertices[2], e1[edge_index])
            mesh.add_edge(vertices[5], vertices[6], e2[edge_index])

        return mesh.add_block(vertices, [n_cells, n_cells, n_cells_axial])

    blocks = [
        block((0, 1, 2, 3)),
        block((0, 4, 5, 1), 1),
        block((1, 5, 6, 2), 2),
        block((2, 6, 7, 3), 3),
        block((3, 7, 4, 0), 0),
    ]

    # inner block: curved edges
    # (could be written in a 'for' loop but is inlined here for clarity)
    ei1 = [rotate(vector(x_1, r_1/2/1.2, 0), -a) for a in e_angles]
    mesh.add_edge(blocks[0].vertices[0], blocks[0].vertices[1], ei1[1])
    mesh.add_edge(blocks[0].vertices[1], blocks[0].vertices[2], ei1[2])
    mesh.add_edge(blocks[0].vertices[2], blocks[0].vertices[3], ei1[3])
    mesh.add_edge(blocks[0].vertices[3], blocks[0].vertices[0], ei1[0])

    ei2 = [rotate(vector(x_2, r_2/2/1.2, 0), -a) for a in e_angles]
    mesh.add_edge(blocks[0].vertices[4], blocks[0].vertices[5], ei2[1])
    mesh.add_edge(blocks[0].vertices[5], blocks[0].vertices[6], ei2[2])
    mesh.add_edge(blocks[0].vertices[6], blocks[0].vertices[7], ei2[3])
    mesh.add_edge(blocks[0].vertices[7], blocks[0].vertices[4], ei2[0])

    # walls are the same for all blocks
    for b in blocks[1:]:
        b.set_patch(['back'], 'wall')

    return blocks

# nozzle
x_start = 0
x_end = l_nozzle
nozzle_blocks = cylinder(x_start, x_end, d_nozzle/2)

# set patches
for b in nozzle_blocks:
    b.set_patch(['right'], 'inlet')    

# divergent cone
x_start = x_end
x_end = x_end + l_cone

cone_blocks = cylinder(x_start, x_end, d_nozzle/2, r_2=d_pipe/2)

# pipe
x_start = x_end
x_end = x_end + l_pipe

pipe_blocks = cylinder(x_start, x_end, d_pipe/2)

for b in pipe_blocks:
    b.set_patch(['left'], 'outlet')
    b.set_cell_size(2, l_cone/cone_blocks[0].n_cells[2], inverse=True)

mesh.write('blockMeshDict.template', 'examples/cylinder/system/blockMeshDict')

# FYI: use transformPoints to rotate this mesh to some other axis
# FYI: use changeDictionary to change 'wall' patch to type 'wall'

# run blockMesh
os.system("blockMesh -case examples/cylinder")