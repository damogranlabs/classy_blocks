from typing import Optional, Set, get_args

from classy_blocks.grading.autograding.params import ChopParams, FixedCountParams, HighReChopParams, SimpleChopParams
from classy_blocks.grading.autograding.probe import Probe
from classy_blocks.grading.chop import Chop
from classy_blocks.items.wires.wire import Wire
from classy_blocks.mesh import Mesh
from classy_blocks.types import ChopTakeType, DirectionType


class GraderBase:
    def __init__(self, mesh: Mesh, params: ChopParams):
        self.mesh = mesh
        self.params = params

        self.mesh.assemble()
        self.probe = Probe(self.mesh)

    def _get_end_size(self, wires: Set[Wire]) -> Optional[float]:
        """Returns average size of wires' last cell"""
        if len(wires) == 0:
            return None

        return sum(wire.grading.end_size for wire in wires) / len(wires)

    def _get_start_size(self, wires: Set[Wire]) -> Optional[float]:
        """Returns average size of wires' first cell"""
        if len(wires) == 0:
            return None

        return sum(wire.grading.start_size for wire in wires) / len(wires)

    def grade_axis(self, axis: DirectionType, take: ChopTakeType) -> None:
        for row in self.probe.get_rows(axis):
            # determine count
            wires = row.get_wires()

            for wire in wires:
                if wire.is_defined:
                    # there's a wire with a defined count already, use that
                    count = wire.grading.count
                    break
            else:
                # take length from a row, as requested
                length = row.get_length(take)
                # and set count from it
                count = self.params.get_count(length)

            for wire in row.get_wires():
                # don't touch defined wires
                # TODO! don't touch wires, defined by USER
                # if wire.is_defined:
                #    # TODO: test
                #    continue

                size_before = self._get_end_size(wire.before)
                size_after = self._get_start_size(wire.after)
                chops = self.params.get_chops(count, wire.length, size_before, size_after)

                wire.grading.clear()
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

    def grade_axis(self, axis, take) -> None:
        for row in self.probe.get_rows(axis):
            # determine count
            wires = row.get_wires()

            for wire in reversed(wires):
                if wire.is_defined:
                    # there's a wire with a defined count already, use that
                    count = wire.grading.count
                    break
            else:
                # take length from a row, as requested
                length = row.get_length(take)
                # and set count from it
                count = self.params.get_count(length)

            for wire in row.get_wires():
                # don't touch defined wires
                # TODO! don't touch wires, defined by USER
                # if wire.is_defined:
                #    # TODO: test
                #    continue

                # make a rudimentary chop first, then adjust
                # in subsequent passes
                chops = [Chop(length_ratio=0.5, count=count // 2), Chop(length_ratio=0.5, count=count // 2)]

                for chop in chops:
                    wire.grading.add_chop(chop)

        super().grade_axis(axis, take)
        super().grade_axis(axis, take)
