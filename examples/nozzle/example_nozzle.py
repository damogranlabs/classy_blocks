#!/usr/bin/python3
import os

import sys
sys.path.append(os.path.abspath("."))

import classy_blocks
from functions import vector, rotate, norm, deg2rad
import numpy as np


# create a nozzle:
# pipe    |   nozzle      |             outside             |        
#                         __________________________________
#                         |                              o |
#                         |                              u |
# _______                 |                              t |
# i       \               |                              l |
# n        \              |                              e |
# l         \_____________|                              t |
# e                                                        |
# t____  __  _____  __  _____  __  _____  __  _____  __  _____  axis of rotation ( = x )


# geometry data:
# wedge angle
alpha = deg2rad(5)

d_pipe = 20e-3
l_pipe = 60e-3

d_nozzle = 3.0e-3
l_nozzle = 15.0e-3
l_cone = 30.0e-3 # contraction

d_outside = 50.0e-3
l_outside = 100.0e-3

# number of cells radially - in nozzle
n_cells_radial = 10
nozzle_cell_size = d_nozzle/2/n_cells_radial

# divide number of cells outside important areas
# and apply grading to match sizes; to save on cell count
cell_dilution = 3

### create a classy mesh
mesh = classy_blocks.Mesh()

def wedge(x_1, x_2, r_inner, r_outer, r_outer_end=None):
    if r_outer_end is None:
        r_outer_end = r_outer

    vertices_front = np.array([
        [x_1, r_inner, r_inner*np.sin(alpha/2)],
        [x_2, r_inner, r_inner*np.sin(alpha/2)],

        [x_2, r_outer_end, r_outer_end*np.sin(alpha/2)],
        [x_1, r_outer, r_outer*np.sin(alpha/2)],
    ])
    vertices_back = np.array([rotate(p, alpha) for p in vertices_front])

    vertices = mesh.add_vertices(np.concatenate((vertices_back, vertices_front)))

    # calculate cell size so that they are approximately square
    radius = min([r_outer - r_inner, r_outer_end - r_inner])
    cell_size = radius/n_cells_radial
    n_cells_axial = int((x_2 - x_1)/cell_size)

    block = mesh.add_block(vertices, [
        n_cells_axial, # x - axial,
        n_cells_radial, # y - radial
        1, # z = 1 - wedge
    ])

    return block, vertices

# 0: inlet
x_start = 0
x_end = l_pipe

block0, _ = wedge(x_start, x_end, 0, d_pipe/2)
block0.set_patch(['front'], 'inlet')
block0.set_patch(['bottom'], 'wall')

# contraction:
x_start = x_end
x_end = x_end + l_cone

block1, vertices_1 = wedge(x_start, x_end, 0, d_pipe/2, r_outer_end=d_nozzle/2)

# you could also add a curved edge for a better nozzle design
# (this point is NOT calculated by any means to produce a good nozzle design,
# it's just a showcase)
x_nozzle_edge = (x_start + x_end)/2
r_nozzle_edge = 0.45*(d_pipe - d_nozzle)/2

mesh.add_edge(vertices_1[2], vertices_1[3], 
    (x_nozzle_edge, r_nozzle_edge, r_nozzle_edge*np.sin(-alpha/2)))

mesh.add_edge(vertices_1[6], vertices_1[7], 
    (x_nozzle_edge, r_nozzle_edge, r_nozzle_edge*np.sin(alpha/2)))


# lower number of cells and apply grading so you don't have too many cells
block1.n_cells[0] = int(block1.n_cells[0]/cell_dilution)
block1.set_cell_size(0, nozzle_cell_size)

block1.set_patch(['bottom'], 'wall')

# nozzle
x_start = x_end
x_end = x_end + l_nozzle

block2, _ = wedge(x_start, x_end, 0, d_nozzle/2)
block2.set_patch(['bottom'], 'wall')

# outside: two blocks, one continued from the nozzle and the other
# one on top of that
x_start = x_end
x_end = x_end + l_outside

block3, _ = wedge(x_start, x_end, 0, d_nozzle/2)
block3.n_cells[0] = int(block3.n_cells[0]/cell_dilution)
block3.set_cell_size(0, nozzle_cell_size, inverse=True)

block3.set_patch(['back'], 'outlet')

block4, _ = wedge(x_start, x_end, d_nozzle/2, d_outside/2)
# 'dilute' cells axially
block4.n_cells[0] = block3.n_cells[0]
block4.set_cell_size(0, nozzle_cell_size, inverse=True)
# and 'dilute' cells radially
block4.n_cells[1] = int((d_outside - d_nozzle)/2 / nozzle_cell_size / cell_dilution)
block4.set_cell_size(1, nozzle_cell_size, inverse=True)

block4.set_patch(['bottom', 'front'], 'wall')
block4.set_patch(['back'], 'outlet')


# all blocks' left and right patches are wedges
for block in [block0, block1, block2, block3, block4]:
    block.set_patch(['left'], 'wedge_left')
    block.set_patch(['right'], 'wedge_right')


mesh.write('blockMeshDict.template', 'examples/nozzle/system/blockMeshDict')

# run blockMesh
os.system("blockMesh -case examples/annulus")