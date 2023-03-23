import abc
import copy
from typing import Optional, TypeVar

from classy_blocks.types import PointType, VectorType

TrBaseT = TypeVar("TrBaseT", bound="TransformableBase")

class TransformableBase(abc.ABC):
    """Base class for every entity that can be transformed;
    the exact maths must be supplied by inheriting objects"""
    @abc.abstractmethod
    def translate(self:TrBaseT, displacement:VectorType) -> TrBaseT:
        """Move by displacement vector; returns the same instance
        to enable chaining of transformations."""

    @abc.abstractmethod
    def rotate(self:TrBaseT, angle:float, axis:VectorType, origin:Optional[PointType]=None) -> TrBaseT:
        """Rotate by 'angle' around 'axis' going through 'origin';
        returns the same instance to enable chaining of transformations."""

    @abc.abstractmethod
    def scale(self:TrBaseT, ratio:float, origin:Optional[PointType]=None) -> TrBaseT:
        """Scale with respect to given origin; returns the same instance
        to enable chaining of transformations. If no origin is given,
        the entity is scaled with respect to its center"""

    def copy(self:TrBaseT) -> TrBaseT:
        """Returns a copy of this object"""
        return copy.deepcopy(self)