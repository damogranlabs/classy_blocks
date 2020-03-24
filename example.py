#!/usr/bin/python3
import os

import classy_blocks
from functions import vector, rotate, norm
import numpy as np

# create a curved duct from four blocks:
# b0: inlet - straight duct
# b1: elbow 1 - 90 degrees elbow, rotated around y-axis
# b2: elbow 2 - 90-degrees elbow, rotated around x-axis
# b3: outlet - straight duct

# geometry data:
d = 0.1 # duct size [m]
l_inlet = 5*d # inlet length
r_elbow = 2*d # radius of elbow center line
l_outlet = 8*d # length of outlet

# auxiliary data: points that define a base for blocks; this will be 'extruded' and 'sweeped'
# see sketch.svg for clarification
d2 = d/2
p_b0 = [ # points that define duct cross-section (start of block 0)
    vector(0,  d2,  d2),
    vector(0,  d2, -d2),
    vector(0, -d2, -d2),
    vector(0, -d2,  d2),
]

### create a classy mesh
mesh = classy_blocks.Mesh()

# a shortcut function: extruded profile
def extrude_block(start_points, extrude_vector, inverse_grading):
    # move start point by a specified vector to get end points
    end_points = [p + extrude_vector for p in start_points]

    # create mesh vertices from points; se Point and Vertice object for
    # explanation of differences;
    # end and start points are reversed so that blockMesh is not whining about
    # blocks being inside-out
    vertices = mesh.add_vertices(end_points + start_points)

    # calculate number of cells;
    # in this case cross-section has a constant number of cells but
    # length-wise cells should get denser towards areas of interest (elbows);
    # use block.set_cell_size() to set first/last cell size to a desired value;
    # grading will be calculated so that the block has the required number of cells
    cell_size = 2*d/10
    n_cells = int(norm(extrude_vector)/cell_size)

    # add a Block object from vertices;
    block = mesh.add_block(vertices, [10, 10, n_cells])
    block.set_cell_size(2, d/10, inverse_grading)

    # assign pathces to block:
    block.set_patch(['top', 'bottom', 'front', 'back'], 'walls')

    return block, end_points

# another shortcut function: rounded block
def sweep_block(start_points, center, axis, angle=np.pi/2):
    # rotate start_points around center for angle
    end_points = [rotate(p - center, angle, axis=axis) + center for p in start_points]
    edge_points = [rotate(p - center, angle/2, axis=axis) + center for p in start_points]

    vertices = mesh.add_vertices(end_points + start_points)

    # add edges: one for the beginning of block and one for the end;
    # see how start_points are passed to this function and where the edges are;
    # if the list of vertices isn't available in current scope but points are,
    # one could also call:
    # mesh.add_edge(
    #   mesh.add_vertex(point_1),
    #   mesh.add_vertex(point_2),
    #   edge_point)
    # )
    # and corresponding vertices will be found instead of creating new, duplicated ones.
    for i in range(4):
        mesh.add_edge(
            vertices[i], # first point
            vertices[i+4], # last point
            edge_points[i]) # point on arc

    elbow_arc_length = abs(angle * (r_elbow-(d/2)))
    cell_size = d/10
    n_cells = int(elbow_arc_length/cell_size)
    block = mesh.add_block(vertices, [10, 10, n_cells])

    # assign patches to block:
    # you don't need (must not!) assign inner patches
    block.set_patch(['top', 'bottom', 'front', 'back'], 'walls')

    return block, end_points

### inlet block
inlet_extrude = vector(l_inlet, 0, 0)
inlet_block, p_b1 = extrude_block(p_b0, inlet_extrude, inverse_grading=True)
inlet_block.set_patch(['left'], 'inlet')

### pipe elbow 1
elbow_rc = vector(l_inlet, 0, r_elbow) # rotation center
_, p_b2 = sweep_block(p_b1, elbow_rc, 'y')

### pipe elbow 2
elbow_rc = vector(l_inlet + r_elbow, r_elbow, r_elbow)
_, p_b3 = sweep_block(p_b2, elbow_rc, 'x')

### outlet
outlet_extrude = vector(0, l_outlet, 0)
outlet_block, _ = extrude_block(p_b3, outlet_extrude, inverse_grading=False)
outlet_block.set_patch(['right'], 'outlet')

mesh.write('blockMeshDict.template', 'example/system/blockMeshDict')

# run blockMesh
os.system("blockMesh -case example")