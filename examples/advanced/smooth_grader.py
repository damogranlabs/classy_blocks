# A terrible blocking of a rectangle for displaying SmoothGrader capabilities

import os

import numpy as np

import classy_blocks as cb

mesh = cb.Mesh()

base = cb.Grid([0, 0, 0], [3, 2, 0], 3, 2)

shape = cb.ExtrudedShape(base, 1)

# turn one block around to test grader's skillz
shape.grid[1][0].rotate(np.pi, [0, 0, 1])

mesh.add(shape)

# move some points to get a mesh with uneven blocks
mesh.assemble()
finder = cb.GeometricFinder(mesh)

move_points = [[0, 1, 1], [2, 1, 1], [3, 1, 1]]

for point in move_points:
    vertex = next(iter(finder.find_in_sphere(point)))
    vertex.translate([0, 0.8, 0])

mesh.set_default_patch("walls", "wall")

grader = cb.SmoothGrader(mesh, 0.05)
grader.grade()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
