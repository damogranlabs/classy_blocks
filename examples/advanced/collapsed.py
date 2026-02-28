import os

import classy_blocks as cb

mesh = cb.Mesh()

bottom_face = cb.Face([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 1, 0]])
top_face = cb.Face([[0, 0, 1], [1, 0.5, 1], [1, 0.5, 1], [0, 1, 1]])

loft = cb.Loft(bottom_face, top_face)
extrude = cb.Extrude(top_face, 1)


revolve_face = cb.Face([[2, 0, 0], [3, 0, 0], [3, 1, 0], [2, 1, 0]])
revolve = cb.Revolve(revolve_face, 1, [1, 0, 0], [0, 0, 0])

mesh.add(loft)
mesh.add(extrude)
mesh.add(revolve)
mesh.assemble()

# grader = cb.FixedCountGrader(mesh)
# mesh.grade()
# mesh.set_default_patch("walls", "wall")

for i in (0, 1, 2):
    loft.chop(i, count=5)
    extrude.chop(i, count=5)
    revolve.chop(i, count=5)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
