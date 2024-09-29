import os

import classy_blocks as cb
from classy_blocks.grading.autograding.grader import SimpleGrader
from classy_blocks.grading.autograding.params import SimpleHighReChopParams

# a cylindrical tank with round end caps
diameter = 0.5
length = 0.5  # including end caps

mesh = cb.Mesh()

cylinder = cb.Cylinder([0, 0, 0], [length, 0, 0], [0, diameter / 2, 0])

wall_name = "tank_wall"

cylinder.set_outer_patch(wall_name)

start_cap = cb.Hemisphere.chain(cylinder, start_face=True)
start_cap.set_outer_patch(wall_name)

end_cap = cb.Hemisphere.chain(cylinder, start_face=False)
end_cap.set_outer_patch(wall_name)

mesh.add(cylinder)
mesh.add(start_cap)
mesh.add(end_cap)

mesh.assemble()

params = SimpleHighReChopParams(0.05)
grader = SimpleGrader(mesh, params)
grader.grade()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
