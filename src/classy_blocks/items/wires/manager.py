
from classy_blocks.items.wires.wire import Wire


class WireManager:
    def __init__(self, wires: list[Wire]):
        self.wires = wires

    def __getitem__(self, index) -> Wire:
        return self.wires[index]

    def __iter__(self):
        return iter(self.wires)

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
    def count(self):
        return self.wires[0].grading.count
