from typing import get_args

from classy_blocks.base.exceptions import InconsistentGradingsError, UndefinedGradingsError
from classy_blocks.cbtyping import ChopTakeType, DirectionType
from classy_blocks.grading.graders.probe import Probe
from classy_blocks.grading.grading import Grading
from classy_blocks.items.block import Block
from classy_blocks.items.wires.axis import Axis
from classy_blocks.items.wires.wire import Wire
from classy_blocks.mesh import Mesh


class ManualGrader:
    """Calculates and distributes user-defined counts/gradings.
    Does not add anything to the mesh - throws an exception if non-graded blocks exist."""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        self.dump = self.mesh.assemble()
        self.dump.block_list.update_lengths()
        self.probe = Probe(self.mesh)

    def _chop_wire(self, wire: Wire, grading: Grading) -> None:
        for coincident in wire.coincidents:
            if coincident.is_graded:
                if coincident.grading.count != grading.count:
                    raise InconsistentGradingsError("Different counts on coincident wires")  # TODO: a nicer message

                # reuse a coincident grading
                wire.grading = coincident.grading
                return

        # set a new grading on this wire
        wire.grading = grading

    def _chop_axis(self, block: Block, direction: DirectionType) -> int:
        """Takes all Chops from block and creates Grading objects for all wires on this axis.
        Returns calculated count if anything was done or 0 if nothing was defined."""
        chops = block.chops.axis_chops[direction]
        if not chops:
            # no chops are defined
            return 0

        axis = block.axes[direction]
        lengths = axis.lengths
        take = chops[0].take
        grading = Grading(0)
        for chop in chops:
            grading.add_chop(chop)

        if take == "avg":
            length = sum(lengths) / 4
        else:
            lengths.sort()

            if take == "max":
                length = lengths[-1]
            else:
                length = lengths[0]

        grading.length = length

        for wire in axis.wires:
            self._chop_wire(wire, grading.copy(wire.length, False))

        return grading.count

    def _distribute_axis(self, axis: Axis) -> bool:
        """Takes an Axis with a defined wire and copies its grading to all non-defined wires of this axis.
        Returns True if the axis is completely graded (all wires), False otherwise."""
        # first, gather defined and non-defined wires
        for wire in axis.wires:
            if wire.is_graded:
                take_from = wire
                break
        else:
            return False

        for wire in axis.wires:
            if not wire.is_graded:
                self._chop_wire(wire, take_from.grading.copy(wire.length, False))

        return True

    def _copy_to_neighbours(self, axis: Axis) -> None:
        for wire in axis.wires:
            if not wire.is_graded:
                continue

            for coincident in wire.coincidents:
                coincident.grading = wire.grading

    def _propagate_axes(self, axes: set[Axis]) -> set[Axis]:
        graded: set[Axis] = set()

        for axis in axes:
            if self._distribute_axis(axis):
                graded.add(axis)

            self._copy_to_neighbours(axis)
            for neighbour in list(axis.neighbours):
                if self._distribute_axis(neighbour):
                    graded.add(neighbour)

        return graded

    def grade(self, _take: ChopTakeType = "avg") -> None:
        # _take isn't needed here - it's defined per-block in chops
        for direction in get_args(DirectionType):
            rows = self.probe.get_rows(direction)

            for row in rows:
                for entry in row.entries:
                    axis_count = self._chop_axis(entry.block, entry.heading)
                    if axis_count != 0:
                        row.set_count(axis_count)

                axes = row.get_axes()
                queue = set(axes)
                graded: set[Axis] = set()
                iteration = 0

                while queue - graded:
                    graded.update(self._propagate_axes(queue))
                    iteration += 1

                    if iteration > len(axes):
                        raise UndefinedGradingsError("")

        # self.dump.block_list.check_consistency()
