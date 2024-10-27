import abc
from typing import get_args

from classy_blocks.grading.autograding.params import ChopParams, FixedCountParams, HighReChopParams, SimpleChopParams
from classy_blocks.grading.autograding.probe import Probe, Row
from classy_blocks.mesh import Mesh
from classy_blocks.types import ChopTakeType, DirectionType


class GraderBase(abc.ABC):
    stages: int

    def __init__(self, mesh: Mesh, params: ChopParams):
        self.mesh = mesh
        self.params = params

        self.mesh.assemble()
        self.probe = Probe(self.mesh)

    def get_count(self, row: Row, take: ChopTakeType) -> int:
        count = row.get_count()

        if count is None:
            # take length from a row, as requested by 'take'
            length = row.get_length(take)
            # and set count from it
            count = self.params.get_count(length)

        return count

    def grade_axis(self, axis: DirectionType, take: ChopTakeType, stage: int) -> None:
        handled_wires = set()

        for row in self.probe.get_rows(axis):
            count = self.get_count(row, take)

            for wire in row.get_wires():
                if wire in handled_wires:
                    continue

                # don't touch defined wires
                # TODO! don't touch wires, defined by USER
                # if wire.is_defined:
                #    # TODO: test
                #    continue

                size_before = wire.size_before
                size_after = wire.size_after

                chops = self.params.get_chops(stage, count, wire.length, size_before, size_after)

                wire.grading.clear()
                for chop in chops:
                    wire.grading.add_chop(chop)

                wire.copy_to_coincidents()

                handled_wires.add(wire)
                handled_wires.update(wire.coincidents)

    def grade(self, take: ChopTakeType = "avg") -> None:
        for axis in get_args(DirectionType):
            for stage in range(self.stages):
                self.grade_axis(axis, take, stage)


class FixedCountGrader(GraderBase):
    """The simplest possible mesh grading: use a constant cell count for all axes on all blocks;
    useful during mesh building and some tutorial cases"""

    stages = 1

    def __init__(self, mesh: Mesh, count: int = 8):
        super().__init__(mesh, FixedCountParams(count))


class SimpleGrader(GraderBase):
    """Simple mesh grading for high-Re cases.
    A single chop is used that sets cell count based on size.
    Cell sizes between blocks differ as blocks' sizes change."""

    stages = 1

    def __init__(self, mesh: Mesh, cell_size: float):
        super().__init__(mesh, SimpleChopParams(cell_size))


class HighReGrader(GraderBase):
    """Parameters for mesh grading for high-Re cases.
    Two chops are added to all blocks; c2c_expansion and and length_ratio
    are utilized to keep cell sizes between blocks consistent
    (as much as possible)"""

    stages = 3

    def __init__(self, mesh: Mesh, cell_size: float):
        super().__init__(mesh, HighReChopParams(cell_size))
