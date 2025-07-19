from typing import Optional

from classy_blocks.base.exceptions import UndefinedGradingsError
from classy_blocks.grading.chop import Chop
from classy_blocks.items.wires.wire import Wire


class WireManager:
    def __init__(self, wires: list[Wire]):
        self.wires = wires
        self.chops: list[Chop] = []

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

    @property
    def undefined(self) -> list[Wire]:
        """Returns a list of wires that have no gradings defined"""
        return [wire for wire in self.wires if not wire.is_defined]

    @property
    def defined(self) -> Optional[Wire]:
        """Returns the first wire with a defined grading"""
        for wire in self.wires:
            if wire.is_defined:
                return wire

        return None

    def propagate_gradings(self) -> None:
        defined = self.defined

        if defined is None:
            raise UndefinedGradingsError("Can't propagate: no defined wires")

        for wire in self.wires:
            # only copy to wires that have no defined coincident wires
            # TODO: test, rethink, refactor, reimplement
            for coincident in wire.coincidents:
                if coincident.is_defined:
                    coincident.copy_to_coincidents()
                    break
            else:
                wire.grading = defined.grading.copy(wire.length, False)
                wire.copy_to_coincidents()

    @property
    def count(self):
        return self.wires[0].grading.count

    @property
    def is_defined(self):
        return all(wire.is_defined for wire in self.wires)
