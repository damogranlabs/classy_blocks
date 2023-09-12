import os

import classy_blocks as cb
from classy_blocks.util import functions as f

mesh = cb.Mesh()

size = 0.1

# The setup is the same as the diffuser_free example
# except that here vertices are only allowed to move
# axially in a straight line.

small_pipe = cb.Cylinder([0, 0, 0], [2, 0, 0], [0, 1, 0])
small_pipe.chop_axial(start_size=size)
small_pipe.chop_radial(start_size=size)
small_pipe.chop_tangential(start_size=size)
mesh.add(small_pipe)

diffuser = cb.Frustum.chain(small_pipe, 0.5, 2)
diffuser.chop_axial(start_size=size)
mesh.add(diffuser)

big_pipe = cb.Cylinder.chain(diffuser, 5)
big_pipe.chop_axial(start_size=size)
mesh.add(big_pipe)

mesh.set_default_patch("walls", "wall")

# Assemble the mesh before making changes
mesh.assemble()

# Find inside vertices
finder = cb.RoundSolidFinder(mesh, diffuser)
inner_vertices = finder.find_core(True)
inner_vertices.update(finder.find_core(False))

# Release those vertices so that optimization can find a better position for them
optimizer = cb.Optimizer(mesh)

for vertex in inner_vertices:
    clamp = cb.LineClamp(vertex, vertex.position, vertex.position + f.vector(1, 0, 0))
    optimizer.release_vertex(clamp)

optimizer.optimize()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
