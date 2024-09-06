from typing import List, Optional

from classy_blocks.base.exceptions import InconsistentGradingsError
from classy_blocks.grading.chop import Chop
from classy_blocks.items.wires.wire import Wire


class WireManager:
    def __init__(self, wires: List[Wire]):
        self.wires = wires
        self.chops: List[Chop] = []

    def __getitem__(self, index) -> Wire:
        return self.wires[index]

    def __iter__(self):
        return iter(self.wires)

    def add_chop(self, chop: Chop) -> None:
        self.chops.append(chop)

    def format_single(self) -> str:
        """Returns a single formatted grading specification for simpleGrading"""
        return self.wires[0].grading.description

    def format_all(self) -> str:
        """Returns all grading specifications, formatted for edgeGrading"""
        return " ".join([wire.grading.description for wire in self.wires])

    @property
    def length(self) -> float:
        """Returns length for each wire of this axis; to be used
        for grading calculation"""
        return sum(wire.edge.length for wire in self.wires) / 4

    @property
    def is_simple(self) -> bool:
        """Returns True if only simpleGrading is required for this axis"""
        # That means all gradings are the same
        first_grading = self.wires[0].grading

        for wire in self.wires[1:]:
            if wire.grading != first_grading:
                return False

        return True

    def check_consistency(self) -> None:
        """Raises an error if not all wires have the same count"""
        counts = [wire.grading.count for wire in self.wires]
        if len(set(counts)) != 1:
            wire_descriptions = [str(wire) for wire in self.wires]
            raise InconsistentGradingsError(f"Inconsistent counts on wires {wire_descriptions} ({counts})")

    @property
    def undefined(self) -> List[Wire]:
        """Returns a list of wires that have no gradings defined"""
        return [wire for wire in self.wires if not wire.is_defined]

    @property
    def defined(self) -> Optional[Wire]:
        """Returns the first wire with a defined grading"""
        for wire in self.wires:
            if wire.is_defined:
                return wire

        return None

    def propagate_gradings(self) -> bool:
        # TODO: update start/end size to match before/after
        defined = self.defined

        if not defined:
            return False

        updated = False

        for wire in self.undefined:
            wire.grading = defined.grading.copy(wire.length, False)
            updated = True

        return updated

    @property
    def count(self):
        return self.wires[0].grading.count

    @property
    def is_defined(self):
        return all(wire.is_defined for wire in self.wires)
