import dataclasses
import functools

from classy_blocks.base.exceptions import InconsistentGradingsError
from classy_blocks.cbtyping import DirectionType
from classy_blocks.construct.edges import Line
from classy_blocks.grading.grading import Grading
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.factory import factory
from classy_blocks.items.vertex import Vertex


@functools.cache
def get_length(wire: "Wire") -> float:
    return wire.edge.length


@dataclasses.dataclass
class WireJoint:
    """Remembers an inline wire (before/after) and
    its orientation (same direction/inverted)."""

    wire: "Wire"
    same_dir: bool

    def __hash__(self):
        return id(self.wire)


class Wire:
    """Represents two vertices that define an edge;
    supplies tools to create and compare, etc"""

    def __init__(self, vertices: list[Vertex], direction: DirectionType, corner_1: int, corner_2: int):
        self.corners = [corner_1, corner_2]
        self.vertices = [vertices[corner_1], vertices[corner_2]]

        self.direction: DirectionType = direction

        # the default edge is 'line' but will be replaced if the user wishes so
        # (that is, not included in edge.factory.registry)
        self.edge: Edge = factory.create(self.vertices[0], self.vertices[1], Line())

        # grading/counts of this wire
        self.grading = Grading(0)

        # multiple wires can be at the same spot; this list holds other
        # coincident wires from different blocks
        self.coincidents: set[Wire] = set()
        # wires that precede this (end with this wire's beginning vertex)
        self.before: set[WireJoint] = set()
        # wires that follow this (start with this wire's end vertex)
        self.after: set[WireJoint] = set()

        self.key = hash(tuple(sorted([v.index for v in self.vertices])))

    @property
    def length(self) -> float:
        return get_length(self)

    def update(self) -> None:
        """Re-sets grading's edge length after the edge has changed"""
        self.grading.length = self.length

    @property
    def is_valid(self) -> bool:
        """A pair with two equal vertices is useless"""
        return self.vertices[0] != self.vertices[1]

    def is_coincident(self, candidate: "Wire") -> bool:
        """Returns True if this wire is in the same spot than the argument,
        regardless of alignment"""
        return self.key == candidate.key

    def is_aligned(self, candidate: "Wire") -> bool:
        """Returns true is this pair has the same alignment
        as the pair in the argument"""
        if not self.is_coincident(candidate):
            raise RuntimeError(f"Wires are not coincident: {self}, {candidate}")

        return self.vertices == candidate.vertices

    def add_coincident(self, candidate: "Wire") -> None:
        """Adds a reference to a coincident wire, if it's aligned"""
        if self.is_coincident(candidate):
            self.coincidents.add(candidate)

    def add_inline(self, candidate: "Wire") -> None:
        """Adds a reference to a wire that is before or after this one
        in the same direction"""
        # this assumes the lines are inline and in the same axis
        # TODO: Test
        # TODO: one-liner, bitte
        if candidate == self:
            return

        if candidate.vertices[1] == self.vertices[0]:
            self.before.add(WireJoint(candidate, True))
        elif candidate.vertices[0] == self.vertices[0]:
            self.before.add(WireJoint(candidate, False))
        elif candidate.vertices[0] == self.vertices[1]:
            self.after.add(WireJoint(candidate, True))
        elif candidate.vertices[1] == self.vertices[1]:
            self.after.add(WireJoint(candidate, False))

    def copy_to_coincidents(self):
        """Copies the grading to all coincident wires"""
        for coincident in self.coincidents:
            if coincident.grading.is_defined:
                continue

            coincident.grading = self.grading.copy(self.length, not coincident.is_aligned(self))

    def check_consistency(self) -> None:
        """Check that coincident wires have the same length and grading"""
        for wire in self.coincidents:
            if wire.length != self.length:
                raise InconsistentGradingsError(f"Coincident wires have different lengths! {self} - {wire}")

            if self.grading != wire.grading:
                raise InconsistentGradingsError(
                    f"Coincident wires have different gradings! {self}:{self.grading} - {wire}:{wire.grading}"
                )

    @property
    def is_defined(self) -> bool:
        return self.grading.is_defined

    def __repr__(self):
        return f"Wire {self.corners[0]}-{self.corners[1]} ({self.vertices[0].index}-{self.vertices[1].index})"

    def __hash__(self):
        return self.key
