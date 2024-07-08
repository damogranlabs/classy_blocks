import os

import classy_blocks as cb

mesh = cb.Mesh()

size = 0.1

# Create a rapidly expanding diffuser that will cause high non-orthogonality
# at the beginning of the contraction; then, move inner vertices so that
# this problem is avoided

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

# Internal edges in Cylinders are Splines by default;
# In this case their endpoints will be moved ('optimized') but not
# the points in between; this will create bad cells. Either re-define these edges after optimization
# or remove them altogether.
diffuser.remove_inner_edges()
small_pipe.remove_inner_edges()
big_pipe.remove_inner_edges()

# Assemble the mesh before making changes
mesh.assemble()

# Find inside vertices (start and stop surfaces of cylinders and frustum);
finder = cb.RoundSolidFinder(mesh, diffuser)
inner_vertices = finder.find_core(True)
inner_vertices.update(finder.find_core(False))

# Release those vertices so that optimization can find a better position for them
optimizer = cb.MeshOptimizer(mesh)

for vertex in inner_vertices:
    clamp = cb.FreeClamp(vertex.position)
    optimizer.release_vertex(clamp)

optimizer.optimize()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
