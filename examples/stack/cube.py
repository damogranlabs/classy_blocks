# a mesh for studying flow over a cube
import os

import classy_blocks as cb

# TODO! direct imports
from classy_blocks.construct.flat.sketches.grid import Grid
from classy_blocks.construct.stack import ExtrudedStack

# cube side;
# it sits in the coordinate system origin
# (center of the cube is [0, 0, side/2])
side = 1

# dimension of blocks
upstream = 3
downstream = 5
width = 3
height = 3

mesh = cb.Mesh()

# first, create cubic blocks all around
point_1 = [-1.5 * side, -1.5 * side, 0]
point_2 = [1.5 * side, 1.5 * side, 0]

base = Grid(point_1, point_2, 3, 3)

stack = ExtrudedStack(base, side * 2, 2)

# add all blocks to mesh
mesh.add(stack)

# fetch the points on appropriate planes and move them to desired dimension
mesh.assemble()
finder = cb.GeometricFinder(mesh)


def move_side(point, normal, axis, value):
    for vertex in finder.find_on_plane(point, normal):
        vertex.position[axis] = value


move_side(point_1, [-1, 0, 0], 0, -upstream * side - side / 2)  # upstream
move_side(point_1, [0, -1, 0], 1, -width * side - side / 2)  # width -y
move_side(point_2, [0, 1, 0], 1, width * side + side / 2)  # width +y
move_side([0, 0, 2 * side], [0, 0, 1], 2, height * side)  # height
move_side(point_2, [1, 0, 0], 0, downstream * side + side / 2)  # downstream

mesh.backport()
mesh.clear()

# Delete the block we're studying;
# TODO: BUG: it matters when this is deleted (but should not be the case?)
mesh.delete(stack.grid[0][1][1])

for shape in stack.shapes:
    for operation in shape.operations:
        for i in range(3):
            operation.chop(i, count=10)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
