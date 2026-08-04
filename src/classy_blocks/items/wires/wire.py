import functools

from classy_blocks.cbtyping import DirectionType
from classy_blocks.construct.edges import Line
from classy_blocks.grading.define.grading import Grading, GradingBase
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.factory import factory
from classy_blocks.items.vertex import Vertex


@functools.cache
def get_length(wire: "Wire") -> float:
    return wire.edge.length


class Wire:
    """Represents two vertices that define an edge;
    supplies tools to create and compare, etc"""

    def __init__(self, vertices: list[Vertex], direction: DirectionType, corner_1: int, corner_2: int):
        self.corners = [corner_1, corner_2]
        self.vertices = [vertices[corner_1], vertices[corner_2]]

        self.direction: DirectionType = direction

        # the default edge is 'line' but will be replaced if the user wishes so
        self.edge: Edge = factory.create(self.vertices[0], self.vertices[1], Line())

        # grading/counts of this wire
        self.grading: GradingBase = Grading(0)

        # multiple wires can be at the same spot; this list holds other
        # coincident wires from different blocks
        self.coincidents: set[Wire] = set()

        # Where this wire is; a naive guess until locate() is called.
        self.canonical = (self.vertices[0].index, self.vertices[1].index)
        self.key: tuple[frozenset[int], frozenset[str]] = (frozenset(self.canonical), frozenset())

    def locate(self, canonical: tuple[int, int], merged_patches: frozenset[str]) -> None:
        """Tells this wire where it *is*, as opposed to which vertices it uses.

        Vertex numbering does not answer that question because face merging duplicates
        vertices on slave patches; 'canonical' holds indexes of unique points instead.
        'merged_patches' are slave patches this whole wire lies within; they go into
        the key so that the two sides of a merged interface stay independent and may
        keep different cell counts. See BlockList.update_neighbours."""
        self.canonical = canonical
        self.key = (frozenset(canonical), merged_patches)

    @property
    def length(self) -> float:
        return get_length(self)

    def update(self) -> None:
        """Re-sets grading's edge length after the edge has changed"""
        self.grading.length = self.length

    def is_coincident(self, candidate: "Wire") -> bool:
        """Returns True if this wire is in the same spot than the argument,
        regardless of alignment"""
        return self.key == candidate.key

    def is_aligned(self, candidate: "Wire") -> bool:
        """Returns true is this pair has the same alignment
        as the pair in the argument"""
        if not self.is_coincident(candidate):
            raise RuntimeError(f"Wires are not coincident: {self}, {candidate}")

        return self.canonical == candidate.canonical

    def add_edge(self, edge: Edge) -> None:
        self.edge = edge

    def add_coincident(self, candidate: "Wire") -> None:
        """Adds a reference to a coincident wire, if it's aligned"""
        if self.is_coincident(candidate):
            self.coincidents.add(candidate)

    @property
    def is_graded(self) -> bool:
        return self.grading.is_defined

    @property
    def is_collapsed(self):
        return self.vertices[0] == self.vertices[1]

    def __repr__(self):
        return f"Wire {self.corners[0]}-{self.corners[1]} ({self.vertices[0].index}-{self.vertices[1].index})"

    # def __hash__(self):
    #     return self.key
