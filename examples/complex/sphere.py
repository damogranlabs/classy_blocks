import os
import classy_blocks as cb

# sphere radius
r = 0.5
l_upstream = 2
l_downstream = 5
width = 2

ball_cell_size = 0.05
domain_cell_size = 2 * ball_cell_size  # maximum cell size
r_prism = 0.75  # radius to which prismatic boundary layers are made
first_layer_thickness = 0.01
expansion_ratio = 1.2

geometry = {
    "inner_sphere": [
        "type   searchableSphere",
        "origin (0 0 0)",
        "centre (0 0 0)",
        f"radius {r}",
    ],
    "outer_sphere": [
        "type   searchableSphere",
        "origin (0 0 0)",
        "centre (0 0 0)",
        f"radius {r_prism}",
    ],
}

# create a 4x4 grid of points;
# source point
co = r_prism / 3**0.5

xc = [-l_upstream, -co, co, l_downstream]
yc = [-width, -co, co, width]
zc = [-width, -co, co, width]

# create a 3x3 grid of blocks; leave the middle out
mesh = cb.Mesh()
oplist = []

projected_faces = {
    4: "top",
    10: "back",
    12: "right",
    14: "left",
    16: "front",
    22: "bottom",
}

for i in range(3):
    for j in range(3):
        for k in range(3):
            if i == j == k == 1:
                # the middle block is the sphere - hollow
                oplist.append(None)
                continue

            n = len(oplist)

            bottom_face = cb.Face(
                [
                    [xc[k], yc[j], zc[i]],
                    [xc[k + 1], yc[j], zc[i]],
                    [xc[k + 1], yc[j + 1], zc[i]],
                    [xc[k], yc[j + 1], zc[i]],
                ]
            )

            top_face = cb.Face(
                [
                    [xc[k], yc[j], zc[i + 1]],
                    [xc[k + 1], yc[j], zc[i + 1]],
                    [xc[k + 1], yc[j + 1], zc[i + 1]],
                    [xc[k], yc[j + 1], zc[i + 1]],
                ]
            )

            loft = cb.Loft(bottom_face, top_face)

            if n in projected_faces:  # blocks around the center
                loft.project_side(projected_faces[n], "outer_sphere", edges=True)

            if k == 0:  # first block - inlet
                loft.set_patch("left", "inlet")

            if k == 2:  # last block - outlet
                loft.set_patch("right", "outlet")

            oplist.append(loft)

# add inner blocks
ci = r / 3**0.5

for i, side in projected_faces.items():
    bottom_face = oplist[i].faces[side]

    if i in (14, 16, 22):
        # starting from block's "other side"
        bottom_face.invert()

    top_points = bottom_face.points * (ci / co)
    top_face = cb.Face(top_points)

    loft = cb.Loft(bottom_face, top_face)
    loft.project_side("top", "inner_sphere", edges=True)

    loft.chop(0, start_size=ball_cell_size)
    loft.chop(1, start_size=ball_cell_size)
    loft.chop(2, start_size=ball_cell_size, end_size=first_layer_thickness)

    loft.set_patch("top", "sphere")
    oplist.append(loft)

# set counts; since count is propagated automatically, only a handful
# of blocks need specific counts set
# x-direction
for i in (12, 14):
    oplist[i].chop(0, start_size=domain_cell_size)

# y-direction
for i in (10, 16):
    oplist[i].chop(1, start_size=domain_cell_size)

# z-direction:
for i in (3, 21):
    oplist[i].chop(2, start_size=domain_cell_size)

for op in oplist:
    if op is not None:
        mesh.add(op)

mesh.set_default_patch("sides", "wall")
mesh.add_geometry(geometry)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
