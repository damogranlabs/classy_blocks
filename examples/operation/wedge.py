import os

import classy_blocks as cb

mesh = cb.Mesh()

# a face with a single bump;
base = cb.Face(
    # points
    [[0, 0, 0], [1, 0, 0], [1, 0.2, 0], [0, 0.2, 0]],
    # edges
    [None, None, cb.Spline([[0.75, 0.15, 0], [0.50, 0.20, 0], [0.25, 0.25, 0]]), None],
)

# move it away from the axis of rotation
# x axis = [1, 0, 0]
base.translate([0, 1, 0])

# create a wedge, then copy it along x-axis,
# representing an annular seal with grooves
wedge = cb.Wedge(base)
wedge.set_outer_patch("static_wall")
wedge.set_inner_patch("rotating_walls")
wedge.chop(0, count=30)

wedges = [wedge.copy().translate([x, 0, 0]) for x in range(1, 6)]
wedges[0].set_patch("left", "inlet")
wedges[-1].set_patch("right", "outlet")

# this will be copied to all next blocks
wedges[0].chop(1, c2c_expansion=1.2, start_size=0.01, invert=True)

# Once an entity is added to the mesh,
# its modifications will not be reflected there;
# adding is the last thing to do
for op in wedges:
    mesh.add(op)

# change patch types and whatnot
mesh.modify_patch("static_wall", "wall")
mesh.modify_patch("rotating_walls", "wall")
mesh.modify_patch("wedge_front", "wedge")
mesh.modify_patch("wedge_back", "wedge")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"))
