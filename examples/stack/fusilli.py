import os

import numpy as np

import classy_blocks as cb
from classy_blocks.base.transforms import Rotation, Translation

# TODO! direct imports
from classy_blocks.construct.flat.sketches.disk import Oval
from classy_blocks.construct.stack import TransformedStack

# Something resembling a pasta.
cell_size = 0.05
oval_point_1 = [0, 0, 0]
oval_point_2 = [0, -1, 0]
oval_radius = 0.5
normal = [0, 0, 1]

mesh = cb.Mesh()

base = Oval(oval_point_1, oval_point_2, normal, oval_radius)

stack = TransformedStack(
    base,
    [Translation([0, 0, 0.3]), Rotation(normal, np.pi / 6, [0, -0.5, 0])],
    12,
    [Translation([0, 0, 0.15]), Rotation(normal, np.pi / 12, [0, -0.5, 0])],
)
stack.shapes[0].chop(0, start_size=cell_size, end_size=cell_size / 10)
stack.shapes[0].chop(1, start_size=cell_size)
stack.chop(count=6)

stack.shapes[0].set_start_patch("inlet")
stack.shapes[-1].set_end_patch("outlet")

mesh.add(stack)

mesh.set_default_patch("walls", "wall")
mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
