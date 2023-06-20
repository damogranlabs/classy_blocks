import os

import classy_blocks as cb

mesh = cb.Mesh()


# Create a 7-block sphere by offsetting a box's faces.
box = cb.Box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])

for i in range(3):
    box.chop(i, count=10)
mesh.add(box)

# faces must point 'out' of the original block or
# newly created blocks will be inside-out;
offset_faces = [box.get_face(orient) for orient in ("bottom", "top", "left", "right", "front", "back")]
for i in (0, 2, 4):
    offset_faces[i].invert()

shell = cb.Shell(offset_faces, 0.5)
shell.chop(count=10)

for operation in shell.operations:
    operation.project_side("top", "sphere", edges=True, points=True)

mesh.add(shell)
mesh.add_geometry(
    {
        "sphere": [
            "type searchableSphere",
            "centre (0 0 0)",
            "radius 1.5",
        ]
    }
)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
