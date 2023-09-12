import os

import classy_blocks as cb

# Example geometry using Loft:
bottom_face = cb.Face(
    # 4 points for face corners
    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
    # edges: arc between 0-1, line between 1-2, arc between 2-3, line between 3-0
    [cb.Arc([0.5, -0.25, 0]), None, cb.Arc([0.5, 1.25, 0]), None],
)

top_face = cb.Face(
    [[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]], [None, cb.Arc([1.25, 0.5, 2]), None, cb.Arc([-0.25, 0.5, 2])]
)

loft = cb.Loft(bottom_face, top_face)
loft.add_side_edge(0, cb.PolyLine([[0.15, 0.15, 0.5], [0.2, 0.2, 1.0], [0.15, 0.15, 1.5]]))  # corners 0 - 4
loft.add_side_edge(1, cb.Arc([0.9, 0.1, 1]))  # 1 - 5
loft.add_side_edge(2, cb.Arc([0.9, 0.9, 1]))  # 2 - 6
loft.add_side_edge(3, cb.Arc([0.1, 0.9, 1]))  # 3 - 7

loft.chop(0, start_size=0.05, c2c_expansion=1.2)
loft.chop(1, count=20)
loft.chop(2, count=30)

mesh = cb.Mesh()
mesh.add(loft)
mesh.set_default_patch("walls", "wall")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"))
