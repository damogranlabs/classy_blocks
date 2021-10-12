#!/usr/bin/env python
import os

import numpy as np

from classy_blocks.classes import block, operations
from classy_blocks.classes.mesh import Mesh
from classy_blocks.util.tools import get_count

# sphere radius
r = 0.5
l_upstream = 2
l_downstream = 5
width = 2

ball_cell_size = 0.05
domain_cell_size = 2*ball_cell_size # maximum cell size
r_prism = 0.75 # radius to which prismatic boundary layers are made
first_layer_thickness = 0.01
expansion_ratio = 1.2

geometry = {
    'inner_sphere': [
        'type   sphere',
        'origin (0 0 0)',
        f'radius {r}',
    ],
    'outer_sphere': [
        'type   sphere',
        'origin (0 0 0)',
        f'radius {r_prism}',
    ]
}

def get_mesh():

    # create a 4x4 grid of points;
    # source point
    co = r_prism / 3**0.5

    xc = [-l_upstream, -co, co, l_downstream]
    yc = [-width,      -co, co, width]
    zc = [-width,      -co, co, width]

    # create a 3x3 grid of blocks; leave the middle out
    m = Mesh()
    oplist = []

    projected_faces = {
        4: 'top',
        10: 'back',
        12: 'right',
        14: 'left',
        16: 'front',
        22: 'bottom',
    }

    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == j == k == 1:
                    # the middle block is the sphere - hollow
                    oplist.append(None)
                    continue

                n = len(oplist)

                bottom_face = operations.Face([
                    [xc[k],   yc[j],   zc[i]],
                    [xc[k+1], yc[j],   zc[i]],
                    [xc[k+1], yc[j+1], zc[i]],
                    [xc[k],   yc[j+1], zc[i]]
                ])

                top_face = operations.Face([
                    [xc[k],   yc[j],   zc[i+1]],
                    [xc[k+1], yc[j],   zc[i+1]],
                    [xc[k+1], yc[j+1], zc[i+1]],
                    [xc[k],   yc[j+1], zc[i+1]],
                ])

                o = operations.Loft(bottom_face, top_face)

                if n in projected_faces: # blocks around the center
                    o.block.project_face(projected_faces[n], 'outer_sphere', edges=True)

                if k == 0: # first block - inlet
                    o.set_patch('left', 'inlet')
                
                if k == 2: # last block - outlet
                    o.set_patch('right', 'outlet')

                m.add(o)
                oplist.append(o)

    # add inner blocks
    ci = r / 3**0.5
    n_bl_cells = get_count(r_prism - r, first_layer_thickness, expansion_ratio)

    for i in projected_faces:
        block = oplist[i].block
        side = projected_faces[i]
        indexes = block.face_map[side]
        
        bottom_points = np.array([block.vertices[ip].point for ip in indexes])
        bottom_face = operations.Face(bottom_points)

        if i in (14, 16, 22):
            # starting from block's "other side"
            bottom_face.invert()

        top_points = bottom_face.points*(ci/co)
        top_face = operations.Face(top_points)

        o = operations.Loft(bottom_face, top_face)
        o.block.project_face('top', 'inner_sphere', edges=True)
        o.count_to_size(0, ball_cell_size)
        o.count_to_size(1, ball_cell_size)

        o.set_cell_count(2, n_bl_cells)
        o.grade_to_size(2, -first_layer_thickness)

        o.set_patch('top', 'sphere')
        m.add(o)

    # set counts; since count is propagated automatically, only a handful
    # of blocks need specific counts set
    # x-direction
    for i in (12, 14):
        oplist[i].count_to_size(0, domain_cell_size)

    # y-direction
    for i in (10, 16):
        oplist[i].count_to_size(1, domain_cell_size)

    # z-direction:
    for i in (3, 21):
        oplist[i].count_to_size(2, domain_cell_size)

    m.set_default_patch('sides', 'wall')

    return m
