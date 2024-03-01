import os

import classy_blocks as cb
from classy_blocks.modify.clamps.free import FreeClamp
from classy_blocks.modify.optimizer import Optimizer

mesh = cb.Mesh()

# generate a cube, consisting of 2x2x2 smaller cubes
for x in (-1, 0):
    for y in (-1, 0):
        for z in (-1, 0):
            box = cb.Box([x, y, z], [x + 1, y + 1, z + 1])

            for axis in range(3):
                box.chop(axis, count=10)

            mesh.add(box)

mesh.set_default_patch("walls", "wall")
mesh.assemble()

# move the middle vertex to a sub-optimal position
finder = cb.GeometricFinder(mesh)
mid_vertex = list(finder.find_in_sphere([0, 0, 0]))[0]
mid_vertex.translate([0.6, 0.6, 0.6])

# find a better spot for the above point using automatic optimization
optimizer = Optimizer(mesh)

# define which vertices can move during optimization, and in which DoF
mid_clamp = FreeClamp(mid_vertex)
optimizer.release_vertex(mid_clamp)


optimizer.optimize()

# mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")

hexmesh = cb.HexMesh(mesh, quality_metrics=True)
hexmesh.write_vtk(os.path.join("simple.vtk"))
