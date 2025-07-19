from typing import get_args

from classy_blocks.cbtyping import ChopTakeType, DirectionType
from classy_blocks.grading.autograding.probe import Probe
from classy_blocks.grading.autograding.row import Row
from classy_blocks.grading.autograding.rules import ChopRules
from classy_blocks.grading.chop import Chop
from classy_blocks.items.wires.wire import Wire
from classy_blocks.mesh import Mesh


class GraderBase:
    def __init__(self, mesh: Mesh, rules: ChopRules):
        self.mesh = mesh
        self.rules = rules

        self.dump = self.mesh.assemble()
        self.dump.block_list.update_lengths()  # TODO: move to a better place
        self.probe = Probe(self.mesh)

    def _chop_wire(self, wire: Wire, chops: list[Chop]) -> None:
        """A shortcut"""
        wire.grading.clear()
        for chop in chops:
            wire.grading.add_chop(chop)

        wire.copy_to_coincidents()

    def check_at_wall(self, row: Row) -> tuple[bool, bool]:
        """Returns True if any block on given row has a wall patch
        (at start and/or end, respectively)."""
        start = False
        end = False

        # Check if there are blocks at the wall;
        for entry in row.entries:
            for wire in entry.wires:
                info = self.probe.get_wire_info(wire)
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

        row.count = self.rules.get_count(length, start_at_wall, end_at_wall)

    def grade_squeezed(self, row: Row) -> None:
        for entry in row.entries:
            for wire in entry.wires:
                if wire.is_defined:
                    continue

                info = self.probe.get_wire_info(wire)

                if self.rules.is_squeezed(row.count, info):
                    chops = self.rules.get_squeezed_chops(row.count, info)
                    self._chop_wire(wire, chops)

    def finalize(self, row: Row) -> None:
        for entry in row.entries:
            for wire in entry.wires:
                if wire.is_defined:
                    continue

                info = self.probe.get_wire_info(wire)
                chops = self.rules.get_chops(row.count, info)

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

        self.dump.block_list.check_consistency()
