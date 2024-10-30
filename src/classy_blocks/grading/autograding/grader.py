import abc
from typing import get_args

from classy_blocks.grading.autograding.params import (
    ChopParams,
    FixedCountParams,
    InflationGraderParams,
    SimpleGraderParams,
    SmoothGraderParams,
)
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
        super().__init__(mesh, SimpleGraderParams(cell_size))


class SmoothGrader(GraderBase):
    """Parameters for mesh grading for high-Re cases.
    Two chops are added to all blocks; c2c_expansion and and length_ratio
    are utilized to keep cell sizes between blocks consistent
    (as much as possible)"""

    stages = 3

    def __init__(self, mesh: Mesh, cell_size: float):
        super().__init__(mesh, SmoothGraderParams(cell_size))


class InflationGrader(GraderBase):
    """Parameters for mesh grading for Low-Re cases.
    To save on cell count, only a required thickness (inflation layer)
    will be covered with thin cells (c2c_expansion in size ratio between them).
    Then a bigger expansion ratio will be applied between the last cell of inflation layer
    and the first cell of the bulk flow.

    Example:
     ________________
    |
    |                 > bulk size (cell_size=bulk, no expansion)
    |________________
    |
    |________________ > buffer layer (c2c = 2)
    |________________
    |================ > inflation layer (cell_size=y+, c2c=1.2)
    / / / / / / / / / wall

    Args:
        first_cell_size (float): thickness of the first cell near the wall
        c2c_expansion (float): expansion ratio between cells in inflation layer
        bl_thickness_factor (int): thickness of the inflation layer in y+ units (relative to first_cell_size)
        buffer_expansion (float): expansion between cells in buffer layer
        bulk_cell_size (float): size of cells inside the domain

        Autochop will take all relevant blocks and choose one to start with - set cell counts
        and other parameters that must stay fixed for all further blocks.
        It will choose the longest/shortest ('max/min') block edge or something in between ('avg').
        The finest grid will be obtained with 'max', the coarsest with 'min'.
    """

    stages = 3

    def __init__(
        self,
        mesh: Mesh,
        first_cell_size: float,
        bulk_cell_size: float,
        c2c_expansion: float = 1.2,
        bl_thickness_factor: int = 30,
        buffer_expansion: float = 2,
    ):
        params = InflationGraderParams(
            first_cell_size, bulk_cell_size, c2c_expansion, bl_thickness_factor, buffer_expansion
        )

        super().__init__(mesh, params)
