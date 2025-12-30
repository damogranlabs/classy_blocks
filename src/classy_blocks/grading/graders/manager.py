from typing import get_args

from classy_blocks.assemble.dump import AssembledDump
from classy_blocks.assemble.settings import Settings
from classy_blocks.base.exceptions import InconsistentGradingsError, UndefinedGradingsError
from classy_blocks.cbtyping import DirectionType
from classy_blocks.grading.analyze.probe import Probe
from classy_blocks.grading.analyze.row import Row
from classy_blocks.grading.define.chop import Chop
from classy_blocks.grading.define.grading import Grading
from classy_blocks.items.block import Block
from classy_blocks.items.wires.axis import Axis
from classy_blocks.items.wires.wire import Wire
from classy_blocks.util.constants import AXIS_PAIRS


class WireGrader:
    def __init__(self, wire: Wire):
        self.wire = wire

    @staticmethod
    def copy(from_wire: Wire, to_wire: Wire) -> None:
        """Copies grading from another wire but takes care to orient it properly"""
        if to_wire.grading.is_defined:
            if to_wire.grading != from_wire.grading:
                raise InconsistentGradingsError(f"Copying grading to a defined wire! (From {from_wire} to {to_wire})")

        to_wire.grading = from_wire.grading.copy(to_wire.length, not to_wire.is_aligned(from_wire))

    def assign(self, grading: Grading) -> None:
        """Assigns a Grading to a wire;
        inherits a grading from a coincident wire if there's one"""
        for coincident in self.wire.coincidents:
            if coincident.is_graded:
                if coincident.grading.count != grading.count:
                    raise InconsistentGradingsError("Different counts on coincident wires")  # TODO: a nicer message

                # reuse a coincident grading
                self.copy(coincident, self.wire)
                return

        # set a new grading on this wire
        self.wire.grading = grading

    def copy_to_coincidents(self, override: bool = False) -> None:
        """Copies this wire's gradings to all coincidents;
        returns False if all coincident wires are defined already"""
        if not self.wire.grading.is_defined:
            raise UndefinedGradingsError(f"Inheriting from a non-graded wire! {self.wire}")

        for coincident in self.wire.coincidents:
            if override:
                coincident.grading.clear()

            if not coincident.is_graded:
                self.copy(self.wire, coincident)

    def re_chop(self, chops: list[Chop]) -> None:
        """Takes a wire that's chopped already and change
        its grading according to a set of different Chops;
        used for edge grading"""
        # remember the wire's count and discard all previous chops;
        # then re-chop using user-specified chops on those wires
        if not self.wire.grading.is_defined:
            raise UndefinedGradingsError(
                "Edge-chopping an un-defined wire; define the mesh fully prior to specifying edge grading"
            )

        wire_count = self.wire.grading.count
        grading = Grading(self.wire.length)

        # add all but the last chop
        for chop in chops[:-1]:
            grading.add_chop(chop)

        # for the last chop, use up the remaining count
        remaining_count = wire_count - grading.count
        if remaining_count < 1:
            raise ValueError(f"Wrong edge grading specification! Remaining count = {remaining_count}")

        # update the chop with remaining count and check if it adds up
        last_chop = chops[-1]

        if last_chop.count is None:
            last_chop.count = remaining_count

        if last_chop.count is not None and last_chop.count != remaining_count:
            raise ValueError(f"Multiple edge chops on count don't add up: {chops[-1].count} != {remaining_count}")

        grading.add_chop(last_chop)

        self.wire.grading = grading

        # replace gradings in coincident wires
        self.copy_to_coincidents(override=True)


class AxisGrader:
    def __init__(self, block: Block, direction: DirectionType):
        self.block = block
        self.direction: DirectionType = direction

    @property
    def axis(self) -> Axis:
        return self.block.axes[self.direction]

    def get_chops(self) -> list[Chop]:
        """Returns chop for this axis only (no edge chops!)"""
        return self.block.chops.axis_chops[self.direction]

    def grade(self) -> int:
        """Takes all Chops from block and creates Grading objects for all wires on this axis.
        Returns calculated count if anything was done or 0 if nothing was defined."""
        chops = self.get_chops()

        if not chops:
            # no chops are defined
            return 0

        axis = self.axis
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
            grader = WireGrader(wire)
            # TODO: assign priority to Chop so it's possible
            # to tell whether to copy the grading to coincidents or take an existing one from them
            grader.assign(grading.copy(wire.length, False))

        return grading.count


class RowGrader:
    def __init__(self, row: Row):
        self.row = row

    def grade(self) -> None:
        for entry in self.row.entries:
            axis_grader = AxisGrader(entry.block, entry.heading)
            axis_count = axis_grader.grade()
            if axis_count != 0:
                # this will raise an exception if called twice with a different count
                self.row.set_count(axis_count)


class GradingDistributor:
    def __init__(self, row: Row):
        self.row = row
        self.axes = set(row.get_axes())

        # each wire belongs to an axis; create a lookup dict
        self.wire_to_axis: dict[Wire, set[Axis]] = {}

        for axis in self.axes:
            for wire in axis.wires:
                if wire not in self.wire_to_axis:
                    self.wire_to_axis[wire] = set()

                self.wire_to_axis[wire].add(axis)
                for coincident in wire.coincidents:
                    if coincident not in self.wire_to_axis:
                        self.wire_to_axis[coincident] = set()

                    self.wire_to_axis[coincident].add(axis)

        self.defined: set[Axis] = set(axis for axis in self.axes if axis.is_graded)

    @property
    def undefined_axes(self) -> set[Axis]:
        return self.axes - self.defined

    @property
    def is_done(self) -> bool:
        return self.defined == self.axes

    def _list_ungraded(self) -> str:
        undefined = self.undefined_axes
        message = "Undefined blocks:\n"

        for entry in self.row.entries:
            if entry.axis in undefined:
                message += f"#{entry.block.index} dir {entry.heading}\n"

        return message

    def _complete_axis(self, axis: Axis) -> bool:
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
                wire.grading = take_from.grading.copy(wire.length, False)

        return True

    def _propagate_wires(self, axis: Axis) -> set[Axis]:
        """Propagates all wires of this axis to their coincidents"""
        fresh_neighbours: set[Axis] = set()

        for wire in axis.wires:
            if not wire.is_graded:
                continue

            wire_grader = WireGrader(wire)
            wire_grader.copy_to_coincidents()
            for coincident in wire.coincidents:
                fresh_neighbours.update(self.wire_to_axis[coincident])

        return fresh_neighbours

    def distribute(self) -> None:
        max_iterations = len(self.axes)
        iteration = 0
        seed_axes = self.defined

        while not self.is_done:
            if iteration > max_iterations:
                raise UndefinedGradingsError("Cannot grade all blocks! " + self._list_ungraded())

            # copy from seeds to their neighbours
            fresh_neighbours: set[Axis] = set()
            for axis in seed_axes:
                fresh_neighbours.update(self._propagate_wires(axis))
            fresh_neighbours -= self.defined

            # the freshly touched axes most probably don't have all the
            # wires defined yet; complete them by copying grading from defined wires
            for axis in fresh_neighbours:
                if self._complete_axis(axis):
                    self.defined.add(axis)

            seed_axes = fresh_neighbours

            iteration += 1


class GradingManager:
    """Calculates and distributes user-defined counts/gradings.
    Does not add anything to the mesh - throws an exception if non-graded blocks exist."""

    def __init__(self, dump: AssembledDump, settings: Settings):
        self.probe = Probe(dump, settings)

    def _grade_row(self, row: Row) -> None:
        grader = RowGrader(row)
        grader.grade()

    def _grade_edges(self, row: Row) -> None:
        # edge grading!
        # find all wires that have edge_chops in ChopCollector and replace
        # edge chops on those wires
        for entry in row.entries:
            chops = entry.block.chops
            if not chops.is_edge_chopped:
                continue

            for pair in AXIS_PAIRS[entry.heading]:
                edge_chops = chops.edge_chops[pair[0]][pair[1]]
                if edge_chops:
                    wire_grader = WireGrader(entry.axis.wires[pair[0]])
                    wire_grader.re_chop(edge_chops)

    def _check_consistency(self, row: Row) -> None:
        axes = row.get_axes()

        count = axes[0].count

        def _raise_count():
            raise InconsistentGradingsError(f"Different counts detected ({count} vs. {axis.count})")

        for axis in row.get_axes():
            if axis.count != count:
                _raise_count()

            for wire in axis.wires:
                if wire.grading.count != count:
                    _raise_count()

                for coincident in wire.coincidents:
                    if wire.grading != coincident.grading:
                        raise InconsistentGradingsError(
                            "Different gradings on coincident wires!"
                            f" ({wire.grading} on {wire} vs. {coincident.grading} on {coincident})"
                        )

    def grade(self):
        for direction in get_args(DirectionType):
            rows = self.probe.get_rows(direction)

            # TODO: benchmark, tests
            # TODO: check check_consistency()!
            for row in rows:
                # convert user-specified axis chops to gradings
                # and set count on this row (must be constant accross the row)
                self._grade_row(row)

            for row in rows:
                # start from graded axes and propagate gradings throughout the row
                distributor = GradingDistributor(row)
                distributor.distribute()

            for row in rows:
                self._grade_edges(row)

            for row in rows:
                self._check_consistency(row)
