import dataclasses
from typing import List, Optional, Set

from classy_blocks.base.exceptions import InconsistentGradingsError
from classy_blocks.construct.edges import Line
from classy_blocks.grading.grading import Grading
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.factory import factory
from classy_blocks.items.vertex import Vertex
from classy_blocks.types import DirectionType


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

    def __init__(self, vertices: List[Vertex], direction: DirectionType, corner_1: int, corner_2: int):
        self.corners = [corner_1, corner_2]
        self.vertices = [vertices[corner_1], vertices[corner_2]]

        self.direction = direction

        # the default edge is 'line' but will be replaced if the user wishes so
        # (that is, not included in edge.factory.registry)
        self.edge: Edge = factory.create(self.vertices[0], self.vertices[1], Line())

        # grading/counts of this wire
        self.grading = Grading(0)

        # multiple wires can be at the same spot; this list holds other
        # coincident wires from different blocks
        self.coincidents: Set[Wire] = set()
        # wires that precede this (end with this wire's beginning vertex)
        self.before: Set[WireJoint] = set()
        # wires that follow this (start with this wire's end vertex)
        self.after: Set[WireJoint] = set()

    @property
    def length(self) -> float:
        return self.edge.length

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
        return self.vertices in [candidate.vertices, candidate.vertices[::-1]]

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
            # TODO: why was this 'if' here?! Investigate and cry.
            # if not coincident.is_defined:
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

    @property
    def size_after(self) -> Optional[float]:
        """Returns average cell size in wires that come after this one (in series/inline);
        None if this is the last wire"""
        # TODO: merge this with size_before somehow
        sum_size: float = 0
        defined_count: int = 0

        for joint in self.after:
            if joint.wire.grading.is_defined:
                defined_count += 1

                if joint.same_dir:
                    sum_size += joint.wire.grading.start_size
                else:
                    sum_size += joint.wire.grading.end_size

        if defined_count == 0:
            return None

        return sum_size / defined_count

    @property
    def size_before(self) -> Optional[float]:
        """Returns average cell size in wires that come before this one (in series/inline);
        None if this is the first wire"""
        # TODO: merge this with size_after somehow
        sum_size: float = 0
        defined_count: int = 0

        for joint in self.before:
            if joint.wire.grading.is_defined:
                defined_count += 1

                if joint.same_dir:
                    sum_size += joint.wire.grading.end_size
                else:
                    sum_size += joint.wire.grading.start_size

        if defined_count == 0:
            return None

        return sum_size / defined_count

    def __repr__(self):
        return f"Wire {self.corners[0]}-{self.corners[1]} ({self.vertices[0].index}-{self.vertices[1].index})"
