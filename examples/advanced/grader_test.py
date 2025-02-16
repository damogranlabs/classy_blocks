import os

import classy_blocks as cb

# a simplified helmholtz nozzle case, made for testing/debugging
# inflation grader.

length = 0.3
r_inner = 0.1
r_outer = 0.2

first_cell_size = 0.01e-3
bulk_cell_size = 1.5e-3


mesh = cb.Mesh()


# nozzle
start = cb.Cylinder([0, 0, 0], [length, 0, 0], [0, r_inner, 0])
start.set_start_patch("inlet")
mesh.add(start)

# chamber: inner cylinder
mid = cb.Cylinder.chain(start, length)
mesh.add(mid)

# chamber outer: expanded ring; the end face will be moved when the mesh is assembled
out = cb.ExtrudedRing.expand(mid, r_outer - r_inner)
mesh.add(out)

# outlet pipe
end = cb.Cylinder.chain(mid, length)
end.set_end_patch("outlet")
mesh.add(end)

mesh.set_default_patch("walls", "wall")

grader = cb.InflationGrader(mesh, first_cell_size, bulk_cell_size)
grader.grade()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
