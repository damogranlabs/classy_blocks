from typing import List, Optional, Set

from classy_blocks.grading.chop import Chop
from classy_blocks.grading.grading import Grading
from classy_blocks.items.wire import Wire
from classy_blocks.types import AxisType


class Axis:
    """One of block axes, numbered 0, 1, 2, and the relevant data"""

    def __init__(self, index: AxisType, wires: List[Wire]):
        self.index = index
        self.wires = wires

        # will be added as blocks are added to mesh
        self.neighbours: Set["Axis"] = set()
        self.chops: List[Chop] = []

        self._grading: Optional[Grading] = None

    def add_neighbour(self, axis: "Axis") -> None:
        """Adds an 'axis' from another block if it shares at least one wire"""
        for this_wire in self.wires:
            for nei_wire in axis.wires:
                if this_wire.is_coincident(nei_wire):
                    self.neighbours.add(axis)

    def is_aligned(self, other: "Axis") -> bool:
        """Returns True if wires of the other axis are aligned
        to wires of this one"""
        # first identify common wires
        for this_wire in self.wires:
            for other_wire in other.wires:
                if this_wire.is_coincident(other_wire):
                    return this_wire.is_aligned(other_wire)

        raise RuntimeError("Axes are not neighbours")

    def chop(self, chop: Chop) -> None:
        """Add a chop to this axis' grading"""
        self.chops.append(chop)

    @property
    def lengths(self) -> List[float]:
        """Returns length for each wire of this axis; to be used
        for grading calculation"""
        return [wire.edge.length for wire in self.wires]

    @property
    def length(self) -> float:
        """Length of block in this axis, according to 'take'
        parameter in the first chop; the default is 'avg' if there
        are no chops yet"""
        if len(self.chops) < 1:
            take = "avg"
        else:
            take = self.chops[0].take

        lengths = self.lengths

        if take == "min":
            return min(lengths)

        if take == "max":
            return max(lengths)

        return sum(lengths) / len(lengths)

    @property
    def is_defined(self) -> bool:
        """Returns True if this axis's counts and gradings are defined"""
        return self.grading.is_defined

    @property
    def grading(self) -> Grading:
        """The grading specification according to current list of chops"""
        if self._grading is None:
            # grading was specified but not yet created;
            # do it now and remember the object
            self._grading = Grading(self.length)

            for chop in self.chops:
                self._grading.add_chop(chop)

        return self._grading

    def copy_grading(self) -> bool:
        """Attempts to copy grading from one of the neighbours;
        returns True if grading has been copied"""
        if self.is_defined:
            # no need to change anything
            return False

        for neighbour in self.neighbours:
            if neighbour.is_defined:
                if self.is_aligned(neighbour):
                    self._grading = neighbour.grading
                else:
                    self._grading = neighbour.grading.inverted

                return True

        return False
