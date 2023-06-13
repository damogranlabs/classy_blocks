import os

import classy_blocks as cb
from classy_blocks.construct.shapes.shell import Shell

mesh = cb.Mesh()

box = cb.Box([0, 0, 0], [1, 1, 1])

for i in range(3):
    box.chop(i, count=10)
mesh.add(box)


offset_faces = [box.get_face("top"), box.get_face("right"), box.get_face("front").invert()]
shell = Shell(offset_faces, 0.2)
shell.chop(count=10)
mesh.add(shell)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
