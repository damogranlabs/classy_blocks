import dataclasses
import abc
import warnings

from classy_blocks.construct.edges import EdgeData
from classy_blocks.items.vertex import Vertex
from classy_blocks.base.element import ElementBase
from classy_blocks.types import EdgeKindType
from classy_blocks.util import functions as f
from classy_blocks.util import constants


@dataclasses.dataclass
class Edge(ElementBase):
    """Common stuff for all edge objects"""

    vertex_1: Vertex
    vertex_2: Vertex
    data: EdgeData

    def __post_init__(self):
        assert isinstance(self.vertex_1, Vertex)
        assert isinstance(self.vertex_2, Vertex)

    @property
    def kind(self) -> EdgeKindType:
        """A shorthand for edge.data.kind"""
        return self.data.kind

    @property
    def is_valid(self) -> bool:
        """Returns True if this edge is elligible to be put into blockMeshDict"""
        if self.data.kind == "line":
            # no need to specify lines
            return False

        # wedge geometries produce coincident
        # edges and vertices; drop those
        if f.norm(self.vertex_1.position - self.vertex_2.position) < constants.TOL:
            return False

        # only arc edges need additional checking (blow-up 1/0 protection)
        # assume others valid
        return True

    @property
    @abc.abstractmethod
    def length(self) -> float:
        """Returns length of this edge's curve"""
        return f.norm(self.vertex_1.position - self.vertex_2.position)

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """string description of the edge to be put in blockMeshDict"""
        # subclasses continue from here
        return f"\t{self.data.kind} {self.vertex_1.index} {self.vertex_2.index} "

    @property
    def center(self):
        warnings.warn("Transforming edge with a default center (0 0 0)!", stacklevel=2)
        return f.vector(0, 0, 0)

    @property
    def parts(self):
        return [self.vertex_1, self.vertex_2, self.data]

    def __eq__(self, other):
        # An Edge is defined between two vertices regardless of
        # its orientation
        return {self.vertex_1.index, self.vertex_2.index} == {other.vertex_1.index, other.vertex_2.index}
