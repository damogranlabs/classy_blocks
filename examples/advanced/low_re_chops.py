import os

import classy_blocks as cb
from classy_blocks.grading.autochop.low_re import LowReChopParams

mesh = cb.Mesh()

start = cb.Box([0, 0, 0], [1, 1, 0.1])

start.chop(0, start_size=0.1)
start.chop(2, count=1)
mesh.add(start)

expand_start = start.get_face("right")
expand = cb.Loft(expand_start, expand_start.copy().translate([1, 0, 0]).scale(2))
expand.chop(2, start_size=0.1)

# HACK: proper usage to be defined
low_re_chops = LowReChopParams(0.01, 0.1)
chops = low_re_chops.get_chops_from_length(1)
for chop in chops:
    expand.chops[1].append(chop)

mesh.add(expand)

contract_start = expand.get_face("top")
contract = cb.Loft(contract_start, contract_start.copy().translate([1, 0, 0]).scale(0.25))
contract.chop(2, start_size=0.1)
mesh.add(contract)

end = cb.Extrude(contract.get_face("top"), 1)
end.chop(2, start_size=0.1)
mesh.add(end)

mesh.set_default_patch("walls", "wall")

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"))
