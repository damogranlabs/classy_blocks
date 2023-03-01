import abc
from typing import Optional

from classy_blocks.types import PointType, VectorType

class TransformableBase(abc.ABC):
    """Base class for every entity that can be transformed;
    the exact maths must be supplied by inheriting objects"""
    @abc.abstractmethod
    def translate(self, displacement:VectorType) -> 'TransformableBase':
        """Move by displacement vector; returns the same instance
        to enable chaining of transformations."""
        
    @abc.abstractmethod
    def rotate(self, angle:float, axis:VectorType, origin:Optional[PointType]=None) -> 'TransformableBase':
        """Rotate by 'angle' around 'axis' going through 'origin';
        returns the same instance to enable chaining of transformations."""
        
    @abc.abstractmethod
    def scale(self, ratio:float, origin:Optional[PointType]=None) -> 'TransformableBase':
        """Scale with respect to given origin; returns the same instance
        to enable chaining of transformations."""
        