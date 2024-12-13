from typing import List, Tuple, get_args

from classy_blocks.grading.autograding.params.base import ChopParams
from classy_blocks.grading.autograding.params.fixed import FixedCountGraderParams
from classy_blocks.grading.autograding.params.inflation import InflationGraderParams
from classy_blocks.grading.autograding.params.simple import SimpleGraderParams
from classy_blocks.grading.autograding.params.smooth import SmoothGraderParams
from classy_blocks.grading.autograding.probe import Probe
from classy_blocks.grading.autograding.row import Row
from classy_blocks.grading.chop import Chop
from classy_blocks.items.wires.wire import Wire
from classy_blocks.mesh import Mesh
from classy_blocks.types import ChopTakeType, DirectionType


class GraderBase:
    """One grader to rule them all. Grading procedure depends on given GraderParams object
    that decides whether to grade a specific block (wire) or to let it pass to

    Behold the most general procedure for grading _anything_.
    For each row in every direction:
    1. Set count
        If there's a wire on the wall - determine 'wall' count (low-re grading etc)
        If not, determine 'bulk' count
        That involves the 'take' keyword so that appropriate block is taken as a reference;
        If there's a wire that has a count, defined by user, use that unconditionally
    2. Chop 'squeezed' blocks
        Where there's not enough space to fit graded cells, use a simple grading
        (or whatever the grader defines)
    3. Chop other blocks
        optionally use multigrading to match neighbours' cell sizes
    """

    def __init__(self, mesh: Mesh, params: ChopParams):
        self.mesh = mesh
        self.params = params

        self.mesh.assemble()
        self.probe = Probe(self.mesh)

    def _chop_wire(self, wire: Wire, chops: List[Chop]) -> None:
        """A shortcut"""
        wire.grading.clear()
        for chop in chops:
            wire.grading.add_chop(chop)

        wire.copy_to_coincidents()

    def check_at_wall(self, row: Row) -> Tuple[bool, bool]:
        """Returns True if any block on given row has a wall patch
        (at start and/or end, respectively)."""
        start = False
        end = False

        # Check if there are blocks at the wall;
        for entry in row.entries:
            for wire in entry.wires:
                # TODO: cache WireInfo
                info = self.probe.get_wire_info(wire, entry.block)
                if info.starts_at_wall:
                    start = True
                if info.ends_at_wall:
                    end = True

        return start, end

    def set_counts(self, row: Row, take: ChopTakeType) -> None:
        if row.count > 0:
            # stuff, pre-defined by the user
            return

        length = row.get_length(take)
        start_at_wall, end_at_wall = self.check_at_wall(row)

        row.count = self.params.get_count(length, start_at_wall, end_at_wall)

    def grade_squeezed(self, row: Row) -> None:
        for entry in row.entries:
            # TODO! don't touch wires, defined by USER
            # if wire.is_defined:
            #    # TODO: test
            #    continue
            for wire in entry.wires:
                if wire.is_defined:
                    continue

                info = self.probe.get_wire_info(wire, entry.block)
                if self.params.is_squeezed(row.count, info):
                    chops = self.params.get_squeezed_chops(row.count, info)
                    self._chop_wire(wire, chops)

    def finalize(self, row: Row) -> None:
        count = row.count

        for entry in row.entries:
            # TODO! don't touch wires, defined by USER
            # if wire.is_defined:
            #    # TODO: test
            #    continue
            for wire in entry.wires:
                if wire.is_defined:
                    continue

                # TODO: cache wire info
                info = self.probe.get_wire_info(wire, entry.block)
                chops = self.params.get_chops(count, info)

                self._chop_wire(wire, chops)

    def grade(self, take: ChopTakeType = "avg") -> None:
        for direction in get_args(DirectionType):
            rows = self.probe.get_rows(direction)
            for row in rows:
                self.set_counts(row, take)
            for row in rows:
                self.grade_squeezed(row)
            for row in rows:
                self.finalize(row)

        self.mesh.block_list.check_consistency()


class FixedCountGrader(GraderBase):
    """The simplest possible mesh grading: use a constant cell count for all axes on all blocks;
    useful during mesh building and some tutorial cases"""

    def __init__(self, mesh: Mesh, count: int = 8):
        super().__init__(mesh, FixedCountGraderParams(count))


class SimpleGrader(GraderBase):
    """Simple mesh grading for high-Re cases.
    A single chop is used that sets cell count based on size.
    Cell sizes between blocks differ as blocks' sizes change."""

    def __init__(self, mesh: Mesh, cell_size: float):
        super().__init__(mesh, SimpleGraderParams(cell_size))


class SmoothGrader(GraderBase):
    """Parameters for mesh grading for high-Re cases.
    Two chops are added to all blocks; c2c_expansion and and length_ratio
    are utilized to keep cell sizes between blocks consistent
    (as much as possible)"""

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
