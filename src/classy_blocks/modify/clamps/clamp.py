import abc
from typing import List

from classy_blocks.items.vertex import Vertex
from classy_blocks.types import NPPointType


class ClampBase(abc.ABC):
    """Movement restriction for optimization by vertex movement"""

    vertex: Vertex

    @property
    @abc.abstractmethod
    def params(self) -> List[float]:
        """List of parameters (Degrees Of Freedom) for this clamp"""
        return []

    @property
    @abc.abstractmethod
    def bounds(self) -> List[List[float]]:
        """Bounds within which to keep optimization parameters"""
        return []

    @abc.abstractmethod
    def update_params(self, params: List[float]) -> None:
        """Update variables, stored in this class with supplied parameters"""

    @property
    @abc.abstractmethod
    def point(self) -> NPPointType:
        """Point according to current parameters"""
