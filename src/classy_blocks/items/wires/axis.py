from classy_blocks.cbtyping import DirectionType
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.wires.manager import WireManager
from classy_blocks.items.wires.wire import Wire


class Axis:
    """One of block axes, indexed 0, 1, 2
    and wires - edges that are defined along the same direction."""

    def __init__(self, direction: DirectionType, wires: list[Wire]):
        self.direction = direction
        self.wires = WireManager(wires)

        # will be added after blocks are added to mesh
        self.neighbours: set[Axis] = set()

    def add_neighbour(self, other: "Axis") -> None:
        """Adds an 'axis' from another block if it shares at least one wire"""
        for this_wire in self.wires:
            for nei_wire in other.wires:
                if this_wire.is_coincident(nei_wire):
                    self.neighbours.add(other)
                    break

    def is_aligned(self, other: "Axis") -> bool:
        """Returns True if wires of the other axis are aligned
        to wires of this one"""
        # first identify common wires
        for this_wire in self.wires:
            for other_wire in other.wires:
                if this_wire.is_coincident(other_wire):
                    return this_wire.is_aligned(other_wire)

        raise RuntimeError("Axes are not neighbours")

    @property
    def lengths(self) -> list[float]:
        return [w.length for w in self.wires]

    @property
    def start_vertices(self) -> set[Vertex]:
        return {wire.vertices[0] for wire in self.wires}

    @property
    def end_vertices(self) -> set[Vertex]:
        return {wire.vertices[1] for wire in self.wires}

    @property
    def count(self) -> int:
        return self.wires.count

    @property
    def is_simple(self) -> bool:
        return self.wires.is_simple

    @property
    def is_graded(self):  # TODO: change to is_chopped
        return all(wire.is_graded for wire in self.wires)

    def __str__(self):
        return f"Axis {self.direction} (" + "|".join(str(wire) for wire in self.wires.wires) + ")"

    def __hash__(self):
        return id(self)
