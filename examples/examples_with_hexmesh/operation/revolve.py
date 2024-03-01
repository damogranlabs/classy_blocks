import os

import classy_blocks as cb
from classy_blocks.util import functions as f

base = cb.Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], [cb.Arc([0.5, -0.2, 0]), None, None, None])

revolve = cb.Revolve(base, f.deg2rad(60), [0, -1, 0], [-2, 0, 0])

# a shortcut for setting count only
revolve.chop(0, count=10)
revolve.chop(1, count=10)
revolve.chop(2, count=30)

mesh = cb.Mesh()
mesh.add(revolve)

mesh.set_default_patch("walls", "wall")
# mesh.write(os.path.join("..", "case", "system", "blockMeshDict"))

hexmesh = cb.HexMesh(mesh, quality_metrics=True)
hexmesh.write_vtk(os.path.join("revolve.vtk"))
