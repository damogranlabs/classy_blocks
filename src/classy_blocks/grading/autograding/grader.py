from typing import get_args

from classy_blocks.grading.autograding.params import ChopParams, FixedCountParams, HighReChopParams, SimpleChopParams
from classy_blocks.grading.autograding.probe import Probe
from classy_blocks.mesh import Mesh
from classy_blocks.types import ChopTakeType, DirectionType


class GraderBase:
    def __init__(self, mesh: Mesh, params: ChopParams):
        self.mesh = mesh
        self.params = params

        self.mesh.assemble()
        self.probe = Probe(self.mesh)

    def grade_axis(self, axis: DirectionType, take: ChopTakeType) -> None:
        for row in self.probe.get_rows(axis):
            length = row.get_length(take)
            count = self.params.get_count(length)

            for wire in row.get_wires():
                chops = self.params.get_chops(count, wire.length)
                for chop in chops:
                    wire.grading.add_chop(chop)

    def grade(self, take: ChopTakeType = "avg") -> None:
        for axis in get_args(DirectionType):
            self.grade_axis(axis, take)


class FixedCountGrader(GraderBase):
    """The simplest possible mesh grading: use a constant cell count for all axes on all blocks;
    useful during mesh building and some tutorial cases"""

    def __init__(self, mesh: Mesh, params: FixedCountParams):
        super().__init__(mesh, params)


class SimpleGrader(GraderBase):
    """Simple mesh grading for high-Re cases.
    A single chop is used that sets cell count based on size.
    Cell sizes between blocks differ as blocks' sizes change."""

    def __init__(self, mesh: Mesh, params: SimpleChopParams):
        super().__init__(mesh, params)


class HighReGrader(GraderBase):
    """Parameters for mesh grading for high-Re cases.
    Two chops are added to all blocks; c2c_expansion and and length_ratio
    are utilized to keep cell sizes between blocks consistent
    (as much as possible)"""

    def __init__(self, mesh: Mesh, params: HighReChopParams):
        super().__init__(mesh, params)
