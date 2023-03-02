import dataclasses
import abc

from typing import Callable, Optional, List

import numpy as np

from classy_blocks.items.vertex import Vertex
from classy_blocks.base.transformable import TransformableBase
from classy_blocks.types import PointType, VectorType
from classy_blocks.util import functions as f
from classy_blocks.util import constants

@dataclasses.dataclass
class Edge(TransformableBase):
    """Common stuff for all edge objects"""
    vertex_1: Vertex
    vertex_2: Vertex

    def __post_init__(self):
        assert isinstance(self.vertex_1, Vertex)
        assert isinstance(self.vertex_2, Vertex)

    @property
    @abc.abstractmethod
    def kind(self) -> str:
        """Edge kind as it is put into blockMeshDict"""

    def transform(self, function: Callable) -> 'Edge':
        """An arbitrary transform of this edge by a specified function"""
        # (only transforms additional points, not vertices;
        # they transform themselves with their own methods)
        return self

    def translate(self, displacement: VectorType):
        """Move all points in the edge (but not start and end)
        by a displacement vector."""
        displacement = np.asarray(displacement, dtype=float)

        return self.transform(lambda p: p + displacement)

    def rotate(self, angle: float, axis: VectorType, origin: Optional[PointType] = None):
        """Rotates all points in this edge (except start and end Vertex) around an
        arbitrary axis and origin (be careful with projected edges, geometry isn't rotated!)"""
        if origin is None:
            origin = [0, 0, 0]

        return self.transform(lambda p: f.arbitrary_rotation(p, axis, angle, origin))

    def scale(self, ratio: float, origin: Optional[PointType] = None) -> 'Edge':
        """Scales the edge points around given origin"""
        return self.transform(lambda p: Vertex.scale_point(p, ratio, origin))

    @property
    @abc.abstractmethod
    def args(self) -> List:
        """Returns arguments that can be used to re-create this
        edge using the factory method"""
        # TODO: test for every edge kind
        return [self.kind]

    @property
    def is_valid(self) -> bool:
        """Returns True if this edge is elligible to be put into blockMeshDict"""
        # TODO: TEST
        if self.kind == 'line':
            # no need to specify lines
            return False

        # wedge geometries produce coincident
        # edges and vertices; drop those
        if f.norm(self.vertex_1.pos - self.vertex_2.pos) < constants.tol:
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
        return f"{self.kind} {self.vertex_1.index} {self.vertex_2.index} "

    def __eq__(self, other):
        # An Edge is defined between two vertices regardless of
        # its orientation
        return {self.vertex_1.index, self.vertex_2.index} == \
            {other.vertex_1.index, other.vertex_2.index}
