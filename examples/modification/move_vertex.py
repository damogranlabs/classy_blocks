import os

import classy_blocks as cb

mesh = cb.Mesh()

cylinder = cb.Cylinder([0, 0, 0], [1, 0, 0], [0, 1, 0])
cylinder.chop_axial(count=10)
cylinder.chop_radial(count=5)
cylinder.chop_tangential(count=8)

mesh.add(cylinder)
mesh.assemble()

finder = cb.GeometricFinder(mesh)
vertex = list(finder.find_in_sphere([1, 0, 0]))[0]
vertex.translate([0.4, 0, 0])

mesh.set_default_patch("walls", "wall")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
