import os
import subprocess

import classy_blocks as cb

mesh = cb.Mesh()

# even if the geometry below is never used in blockMeshDict
# it is still added and checked; its input (the subject of the tests)
# still have to be valid
pnn_plane = cb.SearchablePlanePointAndNormal("plane_1", [0, 0, 0], [0, 0, 1])
ep_plane = cb.SearchablePlaneEmbeddedPoints("plane_2", [1, 0, 0], [0, 0, 1], [1, 1, 1])
s_sphere = cb.SearchableSphere("sphere_1", [0, 0, 0], 2)
s_cylinder = cb.SearchableCylinder("cylinder_1", [0, 0, 0], [1, 0, 0], 1)
s_cone = cb.SearchableCone("cone_1", [0, 0, 0], 2, 1, [1, 0, 0], 1, 0.5)
trisurf = cb.TriSurface("terrain", "terrain.stl")

mesh.add_geometry(pnn_plane)
mesh.add_geometry(ep_plane)
mesh.add_geometry(s_sphere)
mesh.add_geometry(s_cylinder)
# mesh.add_geometry(s_cone) # only on ESI version
mesh.add_geometry(trisurf)

mesh.add(cb.Box([0, 0, 0], [1, 1, 1]))

grader = cb.FixedCountGrader(mesh)
grader.grade()

mesh.write("examples/case/system/blockMeshDict")


os.chdir("examples/case")
subprocess.run("blockMesh", check=True, capture_output=True)
