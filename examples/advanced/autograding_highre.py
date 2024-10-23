import os

import classy_blocks as cb
from classy_blocks.grading.autograding.grader import HighReGrader
from classy_blocks.grading.autograding.params import HighReChopParams

mesh = cb.Mesh()

base = cb.Grid([0, 0, 0], [3, 2, 0], 3, 2)

shape = cb.ExtrudedShape(base, 1)
mesh.add(shape)
mesh.assemble()
finder = cb.GeometricFinder(mesh)

move_points = [[0, 1, 1], [2, 1, 1], [3, 1, 1]]

for point in move_points:
    vertex = list(finder.find_in_sphere(point))[0]
    vertex.translate([0, 0.8, 0])

# TODO! Un-hack
mesh.block_list.update()

mesh.set_default_patch("walls", "wall")


params = HighReChopParams(0.05)
grader = HighReGrader(mesh, params)
grader.grade()

# params = SimpleChopParams(0.05)
# grader = SimpleGrader(mesh, params)
# grader.grade(take="max")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
