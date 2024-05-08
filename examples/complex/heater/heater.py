import os
from typing import List

import numpy as np

import classy_blocks as cb
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.flat.sketches.disk import WrappedDisk
from classy_blocks.construct.operations.operation import Operation

# TODO: direct imports!
from classy_blocks.construct.shape import ExtrudedShape, Shape
from classy_blocks.construct.stack import RevolvedStack
from classy_blocks.util import functions as f

# geometry
heater_diameter = 10
heater_length = 50
bend_radius = 2 * heater_diameter

# cell sizing
solid_cell_size = 1
fluid_cell_size = 0.5
first_cell_size = 0.05
c2c_expansion = 1.2

# mesh parameters (do not edit)
wrapping_height = (bend_radius + heater_diameter / 2) / 1.5

# coordinates, decomposed into grid, a.k.a. 'levels'
xlv = [
    -heater_length,
    0,
    bend_radius - wrapping_height / 2,
    bend_radius - heater_diameter / 2,
    bend_radius,
    bend_radius + heater_diameter / 2,
    bend_radius + wrapping_height / 2,
]

ylv = [
    -bend_radius - wrapping_height / 2,
    -bend_radius - heater_diameter / 2,
    -bend_radius,
    -bend_radius + wrapping_height / 2,
]
zlv = [-wrapping_height / 2, wrapping_height / 2]

heater_start_point = f.vector(xlv[0], ylv[2], 0)
wrapping_corner_point = heater_start_point + f.vector(0, wrapping_height / 2, -wrapping_height / 2)

mesh = cb.Mesh()


def set_cell_zones(shape: Shape):
    solid_indexes = list(range(5))
    fluid_indexes = list(range(5, 9))

    for i in solid_indexes:
        shape.operations[i].set_cell_zone("solid")

    for i in fluid_indexes:
        shape.operations[i].set_cell_zone("fluid")


# Cross-section of heater and fluid around it
WrappedDisk.chops[0] = [1]  # the solid part, chop the fluid zone manually
heater_xs = WrappedDisk(heater_start_point, wrapping_corner_point, heater_diameter / 2, [1, 0, 0])

# The straight part of the heater, part 1: bottom
straight_1 = ExtrudedShape(heater_xs, heater_length)

straight_1.chop(0, start_size=solid_cell_size)
straight_1.chop(1, start_size=solid_cell_size)
straight_1.chop(2, start_size=solid_cell_size)
straight_1.operations[5].chop(0, start_size=first_cell_size, c2c_expansion=c2c_expansion)
set_cell_zones(straight_1)
mesh.add(straight_1)


# The curved part of heater (and fluid around it); constructed from 4 revolves
heater_arch = RevolvedStack(straight_1.sketch_2, np.pi / 4, [0, 0, 1], [0, 0, 0], 4)
heater_arch.chop(start_size=solid_cell_size, take="min")
for shape in heater_arch.shapes:
    set_cell_zones(shape)
mesh.add(heater_arch)


# The straight part of heater, part 2: after the arch
straight_2 = ExtrudedShape(heater_arch.shapes[-1].sketch_2, heater_length)
set_cell_zones(straight_2)
mesh.add(straight_2)

# The arch creates a semi-cylindrical void; fill it with a semi-cylinder, of course
arch_fill = cb.SemiCylinder([0, 0, zlv[0]], [0, 0, zlv[1]], [xlv[1], ylv[3], zlv[0]])

mesh.add(arch_fill)
arch_fill.chop_radial(start_size=solid_cell_size)


class NearestSketch(Sketch):
    def __init__(self, operations: List[Operation], far_point):
        far_point = np.array(far_point)
        self._faces = [op.get_closest_face(far_point) for op in operations]

    @property
    def faces(self):
        return self._faces

    @property
    def grid(self):
        return [self.faces]

    @property
    def center(self):
        return np.average([face.center for face in self.faces], axis=0)


fill_operations = np.take(arch_fill.operations, (0, 1, 2, 5))
cylinder_xs = NearestSketch(fill_operations, [-2 * heater_length, 0, 0])
pipe_fill = ExtrudedShape(cylinder_xs, [-heater_length, 0, 0])
reorienter = cb.ViewpointReorienter([-2 * heater_length, 0, 0], [0, heater_length, 0])
for operation in pipe_fill.operations:
    reorienter.reorient(operation)
mesh.add(pipe_fill)

for operation in [*arch_fill.operations, *pipe_fill.operations]:
    operation.set_cell_zone("fluid")


mesh.write(os.path.join("..", "..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
