import abc
from typing import get_args

from classy_blocks.grading.autograding.params import ChopParams, FixedCountParams, SimpleChopParams
from classy_blocks.grading.autograding.probe import Probe
from classy_blocks.mesh import Mesh
from classy_blocks.types import ChopTakeType, DirectionType


class GraderBase(abc.ABC):
    def __init__(self, mesh: Mesh, params: ChopParams):
        self.mesh = mesh
        self.params = params

        self.probe = Probe(self.mesh)

    @abc.abstractmethod
    def grade(self, take: ChopTakeType = "avg") -> None:
        self.mesh.assemble()


class FixedCountGrader(GraderBase):
    """The simplest possible mesh grading: use a constant cell count for all axes on all blocks;
    useful during mesh building and some tutorial cases"""

    def __init__(self, mesh: Mesh, params: FixedCountParams):
        super().__init__(mesh, params)

    def grade(self, _):
        super().grade()

        # just throw the same count into all blocks and be done
        chops = self.params.get_chops_from_length(0)

        for block in self.mesh.blocks:
            for axis in block.axes:
                axis.chops = chops


class SimpleGrader(GraderBase):
    """Simple mesh grading for high-Re cases.
    A single chop is used that sets cell count based on size.
    Cell sizes between blocks differ as blocks' sizes change."""

    def __init__(self, mesh: Mesh, params: SimpleChopParams):
        super().__init__(mesh, params)

    def grade_axis(self, axis: DirectionType, take: ChopTakeType) -> None:
        for row in self.probe.get_rows(axis):
            length = row.get_length(take)

            chops = self.params.get_chops_from_length(length)

            for block in row.blocks:
                block.axes[axis].chops = chops

    def grade(self, take: ChopTakeType = "avg"):
        super().grade()

        for axis in get_args(DirectionType):
            self.grade_axis(axis, take)
