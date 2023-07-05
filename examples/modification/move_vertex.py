import os

import classy_blocks as cb

mesh = cb.Mesh()

cylinder = cb.Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])
cylinder.chop_axial(count=10)
cylinder.chop_radial(count=5)
cylinder.chop_tangential(count=8)

mesh.add(cylinder)
mesh.assemble()

finder = cb.VertexFinder(mesh)
vertex = finder.by_position([1, 0, 0])[0]
vertex.translate([0.4, 0, 0])

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
