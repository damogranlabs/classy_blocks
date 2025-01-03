# A terrible blocking of a rectangle for displaying InflationGrader capabilities
import os

import numpy as np

import classy_blocks as cb

mesh = cb.Mesh()

base = cb.Grid([0, 0, 0], [3, 2, 0], 3, 2)

shape = cb.ExtrudedShape(base, 1)

# turn one block around to test grader's skillz
shape.grid[0][1].rotate(np.pi, [0, 0, 1])

mesh.add(shape)

mesh.set_default_patch("boundary", "patch")
for i in (0, 2):
    shape.operations[i].set_patch("front", "walls")
shape.operations[1].set_patch("back", "walls")

for i in (3, 4, 5):
    shape.operations[i].set_patch("back", "walls")

for i in (0, 3):
    shape.operations[i].set_patch("left", "walls")

mesh.modify_patch("walls", "wall")

# move some points to get a mesh with uneven blocks
mesh.assemble()
finder = cb.GeometricFinder(mesh)

move_points = [[0, 1, 1], [2, 1, 1], [3, 1, 1]]

for point in move_points:
    vertex = list(finder.find_in_sphere(point))[0]
    vertex.translate([0, 0.8, 0])

grader = cb.InflationGrader(mesh, 5e-3, 0.1)
grader.grade(take="max")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
