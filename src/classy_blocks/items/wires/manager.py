import abc
from typing import List

from classy_blocks.base.exceptions import InconsistentGradingsError
from classy_blocks.grading.chop import Chop
from classy_blocks.grading.grading import Grading
from classy_blocks.items.wires.wire import Wire


class WireManagerBase(abc.ABC):
    def __init__(self, wires: List[Wire]):
        self.wires = wires
        self.chops: List[Chop] = []

    def __getitem__(self, index) -> Wire:
        return self.wires[index]

    def __iter__(self):
        return iter(self.wires)

    def add_chop(self, chop: Chop) -> None:
        self.chops.append(chop)

    @abc.abstractmethod
    def grade(self) -> None:
        """Convert data from user or neighbour to Grading objects on wires"""

    @property
    @abc.abstractmethod
    def is_defined(self) -> bool:
        """Returns True if there's enough data to define a grading on this axis"""

    @property
    @abc.abstractmethod
    def count(self) -> int:
        """Returns cell count"""

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


class WireChopManager(WireManagerBase):
    """Responsible for conversion of user-specified Chops
    into Grading objects for each wire."""

    def __init__(self, wires: List[Wire]):
        super().__init__(wires)

        # Chops and Grading that is done on the whole axis;
        # Cell count is defined here. Other values are a matter of each individual Wire.
        self.grading = Grading(self.length)

    @property
    def is_defined(self) -> bool:
        """True if chops have been delegated to each wire and gradings calculated"""
        if len(self.chops) == 0:
            return False

        # either all wires are defined or none is
        if not self.wires[0].grading.is_defined:
            self.grade()

        return True

    def grade(self) -> None:
        # Create a proper Grading from chops
        for chop in self.chops:
            self.grading.add_chop(chop)

        # Distribute this axis' grading to each wire
        for wire in self.wires:
            # chop wires from specification if it exists
            for chop in self.chops:
                wire.add_chop(chop.copy_preserving())

    @property
    def count(self):
        return self.grading.count


class WirePropagateManager(WireManagerBase):
    """Responsible for finding a neighbour with defined grading and
    distribution of that to other wires on this axis."""

    def __init__(self, wires: List[Wire]):
        super().__init__(wires)

    @property
    def is_defined(self):
        return self.wires[0].grading.is_defined

    def grade(self):
        """Checks each wire whether their coincidents (wires from other blocks)
        have grading defined already; if so, copy it and return True.
        Returns False otherwise"""
        self.copy_neighbours()
        self.propagate_grading()
        self.check_consistency()

    def copy_neighbours(self) -> None:
        """Checks for defined neighbours and copies grading from them"""
        # when using edge grading, numbers must be exactly the same
        # on coincident edges or blockMesh will whine;
        # it's better to just copy them
        for wire in self.wires:
            for coincident in wire.coincidents:
                if coincident.grading.is_defined:
                    if coincident.is_aligned(wire):
                        wire.grading = coincident.grading
                    else:
                        wire.grading = coincident.grading.inverted

    def propagate_grading(self) -> None:
        # take cell count from chops and create gradings on wires that
        # don't yet have one

        # calculate chops by adding them to a fictional Grading object;
        # this will fill the .results dict with useful data
        grading = Grading(self.length)
        for chop in self.chops:
            grading.add_chop(chop)

        for wire in self.wires:
            if not wire.grading.is_defined:
                for chop in self.chops:
                    wire.grading.add_chop(chop)

    def check_consistency(self):
        """Raises an error if not all wires have the same count"""
        counts = [wire.grading.count for wire in self.wires]
        if len(set(counts)) != 1:
            wire_descriptions = [str(wire) for wire in self.wires]
            raise InconsistentGradingsError(f"Inconsistent counts on wires {wire_descriptions} ({counts})")

    @property
    def count(self):
        return sum(chop.results["count"] for chop in self.chops)
