# a mesh for studying flow over a cube
import os

import classy_blocks as cb

# cube side;
# it sits in the coordinate system origin
# (center of the cube is [0, 0, side/2])
side = 1

# dimension of blocks
upstream = 3
downstream = 5
width = 3
height = 3

# cell sizing
wall_size = 0.05
far_size = 0.5

mesh = cb.Mesh()

# first, create cubic blocks all around
point_1 = [-1.5 * side, -1.5 * side, 0]
point_2 = [1.5 * side, 1.5 * side, 0]

base = cb.Grid(point_1, point_2, 3, 3)

stack = cb.ExtrudedStack(base, side * 2, 2)

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

# chop relevant blocks
stack.grid[0][1][0].chop(0, start_size=far_size, end_size=wall_size)
stack.grid[0][1][2].chop(0, start_size=wall_size, end_size=far_size)
stack.grid[1][1][1].chop(0, start_size=wall_size)

stack.grid[0][0][1].chop(1, start_size=far_size, end_size=wall_size)
stack.grid[0][2][1].chop(1, start_size=wall_size, end_size=far_size)
stack.grid[1][1][1].chop(1, start_size=wall_size)

stack.grid[0][0][0].chop(2, start_size=wall_size)
stack.grid[1][1][1].chop(2, start_size=wall_size, end_size=far_size)

# Set patches
for operation in stack.get_slice(0, 0):
    operation.set_patch("left", "inlet")

for operation in stack.get_slice(0, -1):
    operation.set_patch("right", "outlet")

for operation in stack.get_slice(2, 0):
    operation.set_patch("bottom", "floor")

stack.grid[0][1][0].set_patch("right", "cube")
stack.grid[0][0][1].set_patch("back", "cube")
stack.grid[0][2][1].set_patch("front", "cube")
stack.grid[0][1][2].set_patch("left", "cube")
stack.grid[1][1][1].set_patch("bottom", "cube")


# Delete the block we're studying;
# TODO: BUG: it matters when this is deleted (but should not be the case?)
mesh.delete(stack.grid[0][1][1])

mesh.set_default_patch("freestream", "patch")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
