import os

import classy_blocks as cb

mesh = cb.Mesh()

bottom_face = cb.Face([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 1, 0]])
top_face = cb.Face([[0, 0, 1], [1, 0.5, 1], [1, 0.5, 1], [0, 1, 1]])

loft = cb.Loft(bottom_face, top_face)
extrude = cb.Extrude(top_face, 1)

mesh.add(loft)
mesh.add(extrude)

mesh.assemble()

# grader = cb.FixedCountGrader(mesh)
# mesh.grade()
# mesh.set_default_patch("walls", "wall")

for i in (0, 1, 2):
    loft.chop(i, count=5)
    extrude.chop(i, count=5)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
