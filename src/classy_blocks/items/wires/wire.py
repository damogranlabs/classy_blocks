from typing import List, Set

from classy_blocks.base.exceptions import InconsistentGradingsError
from classy_blocks.construct.edges import Line
from classy_blocks.grading.grading import Grading
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.factory import factory
from classy_blocks.items.vertex import Vertex


class Wire:
    """Represents two vertices that define an edge;
    supplies tools to create and compare, etc"""

    def __init__(self, vertices: List[Vertex], axis: int, corner_1: int, corner_2: int):
        self.corners = [corner_1, corner_2]
        self.vertices = [vertices[corner_1], vertices[corner_2]]

        self.axis = axis

        # the default edge is 'line' but will be replaced if the user wishes so
        # (that is, not included in edge.factory.registry)
        self.edge: Edge = factory.create(self.vertices[0], self.vertices[1], Line())

        # grading/counts of this wire
        self.grading = Grading(0)

        # multiple wires can be at the same spot; this list holds other
        # coincident wires from different blocks
        self.coincidents: Set[Wire] = set()
        # wires that precede this (end with this wire's beginning vertex)
        self.before: Set[Wire] = set()
        # wires that follow this (start with this wire's end vertex)
        self.after: Set[Wire] = set()

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

    def is_coincident(self, wire: "Wire") -> bool:
        """Returns True if this wire is in the same spot than the argument,
        regardless of alignment"""
        return self.vertices in [wire.vertices, wire.vertices[::-1]]

    def is_aligned(self, wire: "Wire") -> bool:
        """Returns true is this pair has the same alignment
        as the pair in the argument"""
        if not self.is_coincident(wire):
            raise RuntimeError(f"Wires are not coincident: {self}, {wire}")

        return self.vertices == wire.vertices

    def add_coincident(self, wire):
        """Adds a reference to a coincident wire, if it's aligned"""
        if self.is_coincident(wire):
            self.coincidents.add(wire)

    def add_series(self, wire):
        """Adds a reference to a wire that is before or after this one"""
        if wire.vertices[1] == self.vertices[0]:
            self.before.add(wire)
        elif wire.vertices[0] == self.vertices[1]:
            self.after.add(wire)

    def copy_to_coincidents(self):
        """Copies the grading to all coincident wires"""
        for coincident in self.coincidents:
            if not coincident.is_defined:
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
