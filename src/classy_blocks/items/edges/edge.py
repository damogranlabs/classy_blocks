import dataclasses
import abc

from typing import Optional

from classy_blocks.construct.edges import EdgeData
from classy_blocks.items.vertex import Vertex
from classy_blocks.base.transformable import TransformableBase
from classy_blocks.types import PointType, VectorType, EdgeKindType
from classy_blocks.util import functions as f
from classy_blocks.util import constants


@dataclasses.dataclass
class Edge(TransformableBase):
    """Common stuff for all edge objects"""

    vertex_1: Vertex
    vertex_2: Vertex
    data: EdgeData

    def __post_init__(self):
        assert isinstance(self.vertex_1, Vertex)
        assert isinstance(self.vertex_2, Vertex)

    def translate(self, displacement: VectorType) -> "Edge":
        """Move all points in the edge (but not start and end)
        by a displacement vector."""
        self.data.translate(displacement)
        return self

    def rotate(self, angle: float, axis: VectorType, origin: Optional[PointType] = None):
        """Rotates all points in this edge (except start and end Vertex) around an
        arbitrary axis and origin (be careful with projected edges, geometry isn't rotated!)"""
        self.data.rotate(angle, axis, origin)
        return self

    def scale(self, ratio: float, origin: Optional[PointType] = None) -> "Edge":
        """Scales the edge points around given origin"""
        self.data.scale(ratio, origin)
        return self

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
        if f.norm(self.vertex_1.pos - self.vertex_2.pos) < constants.TOL:
            return False

        # only arc edges need additional checking (blow-up 1/0 protection)
        # assume others valid
        return True

    @property
    @abc.abstractmethod
    def length(self) -> float:
        """Returns length of this edge's curve"""
        return f.norm(self.vertex_1.pos - self.vertex_2.pos)

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """string description of the edge to be put in blockMeshDict"""
        # subclasses continue from here
        return f"\t{self.data.kind} {self.vertex_1.index} {self.vertex_2.index} "

    def __eq__(self, other):
        # An Edge is defined between two vertices regardless of
        # its orientation
        return {self.vertex_1.index, self.vertex_2.index} == {other.vertex_1.index, other.vertex_2.index}
