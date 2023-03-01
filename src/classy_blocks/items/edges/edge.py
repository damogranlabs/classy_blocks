import dataclasses
import abc

from typing import ClassVar, Callable, Optional

import numpy as np

from classy_blocks.items.vertex import Vertex
from classy_blocks.types import PointType, VectorType
from classy_blocks.util import functions as f
from classy_blocks.util import constants

@dataclasses.dataclass
class Edge(abc.ABC):
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