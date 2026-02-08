import os

import classy_blocks as cb
from classy_blocks.construct.edges import Collapsed

mesh = cb.Mesh()

face = cb.Face([[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, 1, 0]], [None, None, Collapsed(), None])
print("Face created")
extrude = cb.Extrude(face, [0, 0, 1])
print("Extrude")

mesh.add(extrude)
mesh.assemble()
print([w.edge.data.kind for w in mesh.blocks[0].wire_list])
print([w.grading.__class__.__name__ for w in mesh.blocks[0].wire_list])

grader = cb.FixedCountGrader(mesh)
mesh.grade()
mesh.set_default_patch("walls", "wall")

# for i in (0, 1, 2):
#     extrude.chop(i, count=5)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
