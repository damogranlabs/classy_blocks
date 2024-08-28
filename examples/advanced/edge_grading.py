import os

import numpy as np

import classy_blocks as cb

mesh = cb.Mesh()

start = cb.Box([0, 0, 0], [1, 1, 0.1])

start.chop(0, start_size=0.1)
start.chop(1, length_ratio=0.5, start_size=0.01, c2c_expansion=1.2, preserve="start_size")
start.chop(1, length_ratio=0.5, end_size=0.01, c2c_expansion=1 / 1.2, preserve="end_size")
start.chop(2, count=1)
mesh.add(start)

expand_start = start.get_face("right")
expand = cb.Loft(expand_start, expand_start.copy().translate([1, 0, 0]).scale(2))
expand.chop(2, start_size=0.1)
mesh.add(expand)

contract_start = expand.get_face("top")
contract = cb.Loft(contract_start, contract_start.copy().translate([1, 0, 0]).scale(0.25))
contract.chop(2, start_size=0.1)
mesh.add(contract)

# rotate the end block to demonstrate grading propagation on non-aligned blocks
end = cb.Extrude(contract.get_face("top"), 1)
end.rotate(np.pi, [0, 0, 1])
end.chop(2, start_size=0.1)

mesh.add(end)

mesh.set_default_patch("walls", "wall")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"))
