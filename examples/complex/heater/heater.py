import os

import numpy as np

import classy_blocks as cb
from classy_blocks.construct.flat.sketches.disk import WrappedDisk

# TODO: direct imports!
from classy_blocks.construct.shape import ExtrudedShape
from classy_blocks.construct.stack import RevolvedStack

wire_diameter = 10
wire_length = 100
bend_radius = 2 * wire_diameter


mesh = cb.Mesh()

heater_xs = WrappedDisk([0, 0, 0], [0, -bend_radius / 2, -bend_radius / 2], wire_diameter / 2, [1, 0, 0])
straight_1 = ExtrudedShape(heater_xs, wire_length)

curve = RevolvedStack(straight_1.sketch_2, np.pi / 4, [0, 0, 1], [wire_length, bend_radius, 0], 4)
for shape in curve.shapes:
    mesh.add(shape)

    for i in range(3):
        for op in shape.operations:
            op.chop(i, count=10)

for i in range(3):
    for op in straight_1.operations:
        op.chop(i, count=10)

mesh.add(straight_1)

mesh.write(os.path.join("..", "..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
