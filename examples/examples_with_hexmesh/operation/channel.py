#!/usr/bin/env python
import os

import numpy as np

import classy_blocks as cb

# A channel with a 90-degred bend, square cross-section, low-Re

# Channel sizing
SCALE = 0.01
WIDTH = 8
HEIGHT = 6
LENGTH_1 = 30
INNER_RADIUS = 5
LENGTH_2 = 15

# Cell sizing
CELL_SIZE = 1
BL_THICKNESS = 0.1
C2C_EXPANSION = 1.2

### Construction
mesh = cb.Mesh()

left_block = cb.Box([0, 0, 0], [LENGTH_1, HEIGHT, WIDTH])
left_block.chop(0, start_size=CELL_SIZE)
left_block.chop(1, length_ratio=0.5, start_size=BL_THICKNESS, c2c_expansion=C2C_EXPANSION)
left_block.chop(1, length_ratio=0.5, end_size=BL_THICKNESS, c2c_expansion=1 / C2C_EXPANSION)
left_block.chop(2, length_ratio=0.5, start_size=BL_THICKNESS, c2c_expansion=C2C_EXPANSION)
left_block.chop(2, length_ratio=0.5, end_size=BL_THICKNESS, c2c_expansion=1 / C2C_EXPANSION)
left_block.set_patch("left", "inlet")
mesh.add(left_block)

revolve_face = left_block.get_face("right")
elbow = cb.Revolve(revolve_face, np.pi / 2, [0, -1, 0], [LENGTH_1, 0, HEIGHT + INNER_RADIUS])
elbow.chop(2, start_size=CELL_SIZE)
mesh.add(elbow)

top_face = elbow.get_face("top")
right_block = cb.Extrude(top_face, LENGTH_2)
right_block.chop(2, start_size=CELL_SIZE)
right_block.set_patch("top", "outlet")
mesh.add(right_block)

mesh.set_default_patch("walls", "wall")
mesh.settings["scale"] = SCALE
# mesh.write(os.path.join("..", "case", "system", "blockMeshDict"))

hexmesh = cb.HexMesh(mesh, quality_metrics=True)
hexmesh.write_vtk(os.path.join("channel.vtk"))
