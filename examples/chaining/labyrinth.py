import os
from typing import List

import classy_blocks as cb
from classy_blocks.grading.autograding.grader import FixedCountGrader
from classy_blocks.grading.autograding.params import SimpleChopParams
from classy_blocks.util import functions as f

mesh = cb.Mesh()

front_point = f.vector(20, -100, 0)
above_point = f.vector(0, 0, 100)
orienter = cb.ViewpointReorienter(front_point, above_point)

extrudes: List[cb.Extrude] = []

face = cb.Face([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
extrude = cb.Extrude(face, 1)
mesh.add(extrude)

for side in ("top", "right", "top", "top", "left", "top"):
    face = extrude.get_face(side)
    # 'left', 'bottom' and 'front' faces' normals point INTO
    # the blocks so the default extrude direction won't work here.
    normal = f.unit_vector(face.center - extrude.center)
    extrude = cb.Extrude(face, normal)

    # Any extrude that was not made from the 'top' face
    # is now oriented differently from the first extrude.
    # To keep things simple and intuitive, let's reorient
    # the operations to keep them consistent - 'top' faces
    # facing upwards.
    orienter.reorient(extrude)
    mesh.add(extrude)

mesh.set_default_patch("walls", "wall")

mesh.assemble()
grader = FixedCountGrader(mesh, SimpleChopParams())
grader.grade()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
