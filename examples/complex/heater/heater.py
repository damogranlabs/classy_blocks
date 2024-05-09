import os
from typing import List

import numpy as np
import parameters as p

import classy_blocks as cb
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.flat.sketches.disk import WrappedDisk
from classy_blocks.construct.operations.operation import Operation

# TODO: direct imports!
from classy_blocks.construct.shape import ExtrudedShape, Shape
from classy_blocks.construct.stack import RevolvedStack

mesh = cb.Mesh()

# some shortcuts
xlv = p.xlv
ylv = p.ylv
zlv = p.zlv


def set_cell_zones(shape: Shape):
    solid_indexes = list(range(5))
    fluid_indexes = list(range(5, 9))

    for i in solid_indexes:
        shape.operations[i].set_cell_zone("solid")

    for i in fluid_indexes:
        shape.operations[i].set_cell_zone("fluid")


# Cross-section of heater and fluid around it
WrappedDisk.chops[0] = [1]  # the solid part, chop the fluid zone manually
heater_xs = WrappedDisk(p.heater_start_point, p.wrapping_corner_point, p.heater_diameter / 2, [1, 0, 0])

# The straight part of the heater, part 1: bottom
straight_1 = ExtrudedShape(heater_xs, p.heater_length)

straight_1.chop(0, start_size=p.solid_cell_size)
straight_1.chop(1, start_size=p.solid_cell_size)
straight_1.chop(2, start_size=p.solid_cell_size)
straight_1.operations[5].chop(0, start_size=p.first_cell_size, c2c_expansion=p.c2c_expansion)
set_cell_zones(straight_1)
mesh.add(straight_1)


# The curved part of heater (and fluid around it); constructed from 4 revolves
heater_arch = RevolvedStack(straight_1.sketch_2, np.pi / 4, [0, 0, 1], [0, 0, 0], 4)
heater_arch.chop(start_size=p.solid_cell_size, take="min")
for shape in heater_arch.shapes:
    set_cell_zones(shape)
mesh.add(heater_arch)


# The straight part of heater, part 2: after the arch
straight_2 = ExtrudedShape(heater_arch.shapes[-1].sketch_2, p.heater_length)
set_cell_zones(straight_2)
mesh.add(straight_2)

# The arch creates a semi-cylindrical void; fill it with a semi-cylinder, of course
arch_fill = cb.SemiCylinder([0, 0, zlv[0]], [0, 0, zlv[1]], [xlv[1], ylv[4], zlv[0]])

mesh.add(arch_fill)
arch_fill.chop_radial(start_size=p.solid_cell_size)


# A custom sketch that takes the closest faces from given operations;
# They will definitely be wrongly oriented but we'll sort that out later
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


cylinder_xs = NearestSketch([arch_fill.operations[i] for i in (0, 1, 2, 5)], [-2 * p.heater_length, 0, 0])
pipe_fill = ExtrudedShape(cylinder_xs, [-p.heater_length, 0, 0])

# reorient the operations in the shape
reorienter = cb.ViewpointReorienter([-2 * p.heater_length, 0, 0], [0, p.heater_length, 0])
for operation in pipe_fill.operations:
    reorienter.reorient(operation)
mesh.add(pipe_fill)

# Domain: offset outermost faces of outermost operations
offset_shapes = [straight_1, *heater_arch.shapes, straight_2]
offset_ops = [shape.grid[2][2] for shape in offset_shapes]
offset_faces = [op.get_face("right") for op in offset_ops]
domain_shell = cb.Shell(offset_faces, ylv[1] - ylv[0])
domain_shell.operations[1].chop(2, start_size=2 * p.fluid_cell_size, end_size=10 * p.fluid_cell_size)
mesh.add(domain_shell)

# The offset faces create a domain that is not rectangular.
# Assemble the mesh and move vertices, then backport the changes so that
# they will be reflected in the offset shapes.
mesh.assemble()

# Vertex indexes are taken from debug.vtk file written right here.
# This is the quickest and simplest method and works as long as blocking
# doesn't change. If there was a parameter that controlled number of blocks
# (for instance, number of levels in the arch stack), then
# we'd have to obtain vertices programatically using GeometricFinder or similar.
for index in (106, 107):
    mesh.vertices[index].position[0] = xlv[7]
    mesh.vertices[index].position[1] = ylv[0]

for index in (110, 111):
    mesh.vertices[index].position[0] = xlv[7]
    mesh.vertices[index].position[1] = -ylv[0]

mesh.backport()
mesh.clear()

# Add blocks to fluid zone (those that haven't been added yet)
for operation in [*arch_fill.operations, *pipe_fill.operations, *domain_shell.operations]:
    operation.set_cell_zone("fluid")

mesh.write(os.path.join("..", "..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
