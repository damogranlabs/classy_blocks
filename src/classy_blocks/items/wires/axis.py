from typing import List, Set

from classy_blocks.grading.chop import Chop
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.manager import WireChopManager, WireManagerBase, WirePropagateManager
from classy_blocks.items.wires.wire import Wire
from classy_blocks.types import AxisType

# Edge grading
# Axis holds 4 wires, that is, edges that are defined along the same direction.
# Wire is an object that holds data about the edge: Edge definition, grading, coincident wires etc.
# Chop object holds all parameters for setting cell count and total expansion ratio
#   but nothing more. An Axis/Wire can be chopped multiple times in case of multigrading, thus
#   having multiple chop objects.
# Grading has a defined length (of an edge) and converts a list of Chops to actual
#   count/total_expansion numbers for meshing.

# A block can be chopped by user input or chops can be copied from neighbour.

# When a block is chopped by the user, a Chop is added to that axis.
# When mesh.write() is called, all chops are fed into a Grading object that
# calculates cell counts by taking the average length of all wires.
# Each chop is now copied to each wire, taking into account a fixed cell count (from chop.results)
# and desired 'keep' parameter to preserve. When a Grading is calculated from chop,
# each corresponding wire's length is taken.

# When block has no chops added, neighbours are checked for defined Chops.
# If chops are defined, so must be gradings. Wires that are coincident copy exactly the same
# grading (or inverted). Other wires within the same block will


class Axis:
    """One of block axes, numbered 0, 1, 2, and the relevant data"""

    def __init__(self, index: AxisType, wires: List[Wire]):
        self.index = index

        # Most blocks/axes will not have user-specified chops;
        # they will usually be copied from neighbours.
        # Some of them, however, will - in those cases PropagateManager
        # will be replaced by a ChopManager
        self.wires: WireManagerBase = WirePropagateManager(wires)

        # will be added after blocks are added to mesh
        self.neighbours: Set[Axis] = set()

    def add_neighbour(self, axis: "Axis") -> None:
        """Adds an 'axis' from another block if it shares at least one wire"""
        for this_wire in self.wires:
            for nei_wire in axis.wires:
                if this_wire.is_coincident(nei_wire):
                    self.neighbours.add(axis)

    def add_sequential(self, axis: "Axis") -> None:
        """Adds an axis that comes before/after this one"""
        # As opposed to neighbours that are 'around' this axis
        if self.start_vertices == axis.end_vertices or self.end_vertices == axis.start_vertices:
            for this_wire in self.wires:
                for nei_wire in axis.wires:
                    this_wire.add_series(nei_wire)

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
        # replace the manager on the first added chop
        if not isinstance(self.wires, WireChopManager):
            self.wires = WireChopManager(self.wires.wires)

        self.wires.add_chop(chop)

    @property
    def is_defined(self) -> bool:
        """Returns True if this axis's counts and gradings are defined"""
        return self.wires.is_defined

    def copy_grading(self) -> bool:
        """Attempts to copy grading from one of the neighbours;
        returns True if grading has been copied"""
        if self.is_defined:
            # no need to change anything
            return False

        for neighbour in self.neighbours:
            if neighbour.is_defined:
                if neighbour.is_aligned(self):
                    for chop in neighbour.wires.chops:
                        self.wires.add_chop(chop.copy_preserving())
                else:
                    for chop in reversed(neighbour.wires.chops):
                        self.wires.add_chop(chop.copy_preserving(inverted=True))

                self.grade()
                return True

        return False

    @property
    def start_vertices(self) -> Set[Vertex]:
        return {wire.vertices[0] for wire in self.wires}

    @property
    def end_vertices(self) -> Set[Vertex]:
        return {wire.vertices[1] for wire in self.wires}

    def grade(self) -> None:
        self.wires.grade()

    @property
    def count(self) -> int:
        return self.wires.count

    @property
    def is_simple(self) -> bool:
        return self.wires.is_simple

    def check_consistency(self) -> None:
        self.wires.check_consistency()
