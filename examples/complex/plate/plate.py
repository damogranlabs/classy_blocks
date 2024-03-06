#!/usr/bin/env python
import os

import parameters as ps

import classy_blocks as cb
from classy_blocks.util import functions as f

# Example: flow over a flat plate of


points = [
    [-ps.pl_thickness, 0, 0],  # 0
    [0, 0, 0],  # 1 (set later)
    [0, ps.pl_thickness, 0],  # 2
    [ps.pl_length, ps.pl_thickness, 0],  # 3
    [ps.pl_length, 0, 0],  # 4
    [-ps.bl_thickness - ps.pl_thickness, 0, 0],  # 5
    [0, 0, 0],  # 6 (set later)
    [0, ps.pl_thickness + ps.bl_thickness, 0],  # 7
    [ps.pl_length, ps.pl_thickness + ps.bl_thickness, 0],  # 8
    [-ps.dm_upstream, 0, 0],  # 9
    [0, 0, 0],  # 10 (later)
    [-ps.dm_upstream, ps.dm_height, 0],  # 11
    [0, 0, 0],  # 12 (l8r)
    [0, ps.dm_height, 0],  # 13
    [ps.pl_length, ps.dm_height, 0],  # 14
]

points[1] = f.rotate(points[0], -f.deg2rad(45), [0, 0, 1], [0, 0, 0])
points[6] = f.rotate(points[5], -f.deg2rad(45), [0, 0, 1], [0, 0, 0])
points[10] = [points[9][0], points[6][1], 0]
points[12] = [points[6][0], points[11][1], 0]

mesh = cb.Mesh()


def make_extrude(indexes):
    face = cb.Face([points[i] for i in indexes])
    op = cb.Extrude(face, ps.z)

    op.chop(2, count=1)

    mesh.add(op)
    return op


extrudes = [
    make_extrude([0, 1, 6, 5]),
    make_extrude([1, 2, 7, 6]),
    make_extrude([2, 3, 8, 7]),
    make_extrude([9, 5, 6, 10]),
    make_extrude([10, 6, 12, 11]),
    make_extrude([6, 7, 13, 12]),
    make_extrude([7, 8, 14, 13]),
]

for index in (0, 1):
    for corner in (0, 2):
        extrudes[index].bottom_face.add_edge(corner, cb.Origin([0, 0, 0]))
        extrudes[index].top_face.add_edge(corner, cb.Origin([0, 0, ps.z]))

for index in (0, 1):
    extrudes[index].chop(0, start_size=ps.cell_size)
extrudes[2].chop(1, start_size=ps.first_cell_thickness, c2c_expansion=ps.c2c_expansion)
extrudes[2].chop(0, start_size=ps.cell_size, end_size=ps.cell_size * ps.max_ar)

extrudes[3].chop(0, start_size=ps.cell_size)
extrudes[5].chop(1, start_size=ps.cell_size)


mesh.write(os.path.join("..", "..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
