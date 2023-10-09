import abc
import dataclasses
import warnings

from classy_blocks.base.element import ElementBase
from classy_blocks.base.exceptions import EdgeCreationError
from classy_blocks.construct.edges import EdgeData
from classy_blocks.items.vertex import Vertex
from classy_blocks.types import EdgeKindType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


@dataclasses.dataclass
class Edge(ElementBase):
    """Common stuff for all edge objects"""

    vertex_1: Vertex
    vertex_2: Vertex
    data: EdgeData

    def __post_init__(self):
        if not (isinstance(self.vertex_1, Vertex) and isinstance(self.vertex_2, Vertex)):
            raise EdgeCreationError(
                "Unable to create `Edge`: at least one of given points is not `Vertex` type",
                f"Vertex 1: {type(self.vertex_1)}, vertex 2: {type(self.vertex_2)}",
            )

    @property
    def kind(self) -> EdgeKindType:
        """A shorthand for edge.data.kind"""
        return self.data.kind

    @property
    def representation(self) -> EdgeKindType:
        """A string that goes into blockMesh"""
        return self.data.kind

    @property
    def is_valid(self) -> bool:
        """Returns True if this edge is elligible to be put into blockMeshDict"""
        if self.kind == "line":
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
        return f"\t{self.representation} {self.vertex_1.index} {self.vertex_2.index} "

    @property
    def center(self):
        warnings.warn("Transforming edge with a default center (0 0 0)!", stacklevel=2)
        return f.vector(0, 0, 0)

    @property
    def parts(self):
        return [self.vertex_1, self.vertex_2, self.data]
