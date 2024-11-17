import os

import numpy as np

import classy_blocks as cb
from classy_blocks.grading.autograding.grader import InflationGrader

mesh = cb.Mesh()

base = cb.Grid([0, 0, 0], [3, 2, 0], 3, 2)

shape = cb.ExtrudedShape(base, 1)

# turn one block around to test grader's skillz
shape.grid[1][0].rotate(np.pi, [0, 0, 1])

mesh.add(shape)

mesh.set_default_patch("boundary", "patch")
for i in (0, 1, 2):
    shape.operations[i].set_patch("front", "walls")
mesh.modify_patch("walls", "wall")

# move some points to get a mesh with uneven blocks
mesh.assemble()
finder = cb.GeometricFinder(mesh)

move_points = [[0, 1, 1], [2, 1, 1], [3, 1, 1]]

for point in move_points:
    vertex = list(finder.find_in_sphere(point))[0]
    vertex.translate([0, 0.8, 0])


# TODO: Hack! mesh.assemble() won't work here but wires et. al. must be updated
mesh.block_list.update()

grader = InflationGrader(mesh, 1e-3, 0.1)
grader.grade()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
