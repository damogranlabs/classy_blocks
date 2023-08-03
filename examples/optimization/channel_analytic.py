import os

import numpy as np

import classy_blocks as cb

mesh = cb.Mesh()

blocks = 5
length = 3  # must be an integer
height = 1
thickness = 1.0
cell_size = 0.1


# create a channel from three blocks;
# the top wall is straight but the bottom is sinusoidal,
# defined by an analytic curve
def shapefunc_front(t):
    return [t, 0.8 * height * np.sin(t * np.pi / length), 0]


def shapefunc_back(t):
    return [t, 0.8 * height * np.cos(t * np.pi / length), -thickness]


# create three lofts side by side; top face is flat, the bottom is created from shape functions
x_coords = np.linspace(0, length, num=blocks, endpoint=True)
boxes = []

for i, x in enumerate(x_coords[:-1]):
    t_1 = x
    t_2 = x_coords[i + 1]

    bottom_face = cb.Face([shapefunc_front(t_1), shapefunc_front(t_2), shapefunc_back(t_2), shapefunc_back(t_1)])

    top_face = cb.Face([[t_1, height, 0], [t_2, height, 0], [t_2, height, -thickness], [t_1, height, -thickness]])

    boxes.append(cb.Loft(bottom_face, top_face))

boxes[0].chop(0, start_size=cell_size)
boxes[0].chop(2, start_size=cell_size)
boxes[0].chop(1, count=1)

for box in boxes[1:]:
    box.chop(0, start_size=cell_size)

for box in boxes:
    mesh.add(box)

mesh.set_default_patch("walls", "wall")

# Assign bottom vertices to shape function;
# clamps will also move vertices to closest points on curve.
# Move top vertices along top edge (linear)
mesh.assemble()
finder = cb.VertexFinder(mesh)
optimizer = cb.Optimizer(mesh)

vertices_front = []
vertices_back = []
vertices_top = []
for x in x_coords[1:-1]:
    vertices_front += finder.by_position([x, 0, 0])
    vertices_back += finder.by_position([x, 0, -thickness])
    vertices_top += finder.by_position([x, height, 0])
    vertices_top += finder.by_position([x, height, -thickness])

for vertex in vertices_front:
    clamp = cb.AnalyticCurveClamp(vertex, shapefunc_front)
    optimizer.release_vertex(clamp)

for vertex in vertices_back:
    clamp = cb.AnalyticCurveClamp(vertex, shapefunc_back)
    optimizer.release_vertex(clamp)

for vertex in vertices_top:
    clamp = cb.LineClamp(vertex, vertex.position, vertex.position + np.array([1, 0, 0]))
    optimizer.release_vertex(clamp)

optimizer.optimize()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
