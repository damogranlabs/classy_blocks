import abc

from typing import Optional, List

from classy_blocks.types import PointType, VectorType

class TransformableBase(abc.ABC):
    """Base class for every entity that can be transformed;
    all transformations work on supplied Point/Vertex and Edge/EdgeData objects"""
    @property
    @abc.abstractmethod
    def movable_entities(self) -> List['TransformableBase']:
        """Points/Vertices and Edge/EdgeData objects
        that define this entity and support transformations;
        these objects will be transformed when moving/rotating/scaling a transformable
        object"""
    
    def translate(self, displacement:VectorType) -> 'TransformableBase':
        """Move by displacement vector; returns the same instance
        to enable chaining of transformations."""
        for entity in self.movable_entities:
            entity.translate(displacement)

        return self

    def rotate(self, angle:float, axis:VectorType, origin:Optional[PointType]=None) -> 'TransformableBase':
        """Rotate by 'angle' around 'axis' going through 'origin';
        returns the same instance to enable chaining of transformations."""
        for entity in self.movable_entities:
            entity.rotate(angle, axis, origin)

        return self

    def scale(self, ratio:float, origin:Optional[PointType]=None) -> 'TransformableBase':
        """Scale with respect to given origin; returns the same instance
        to enable chaining of transformations."""
        for entity in self.movable_entities:
            entity.scale(ratio, origin)

        return self
