import os

import numpy as np

import classy_blocks as cb

mesh = cb.Mesh()

# A torus; a simple demonstration of copying and transforming
inner_radius = 0.3
outer_radius = 0.6

n_segments = 8
n_cells = 5


sweep_angle = 2 * np.pi / n_segments

elbow = cb.Elbow(
    [inner_radius + (outer_radius - inner_radius) / 2, 0, 0],
    [inner_radius, 0, 0],
    [0, -1, 0],
    -sweep_angle,
    [0, 0, 0],
    [0, 0, 1],
    (outer_radius - inner_radius) / 2,
)

# counts and gradings
elbow.chop_tangential(count=n_cells)
elbow.chop_radial(count=n_cells)
elbow.chop_axial(count=n_cells)
mesh.add(elbow)

for i in range(1, n_segments):
    segment = elbow.copy().rotate(-i * sweep_angle, [0, 0, 1], [0, 0, 0])
    mesh.add(segment)

mesh.set_default_patch("walls", "wall")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
