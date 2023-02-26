from typing import Optional, List

from classy_blocks.types import PointType, VectorType

class TransformableBase:
    """Base class for every entity that can be transformed;
    all transformations work on supplied Point/Vertex and Edge/EdgeData objects"""
    # Points/Vertices and Edge/EdgeData objects
    #  that define this entity and support transformations
    points:List['TransformableBase']
    edges:Optional[List['TransformableBase']] = None
    
    def translate(self, displacement:VectorType) -> 'TransformableBase':
        """Move by displacement vector; returns the same instance
        to enable chaining of transformations."""
        for point in self.points:
            point.translate(displacement)

        if self.edges is not None:
            for edge in self.edges:
                edge.translate(displacement)

        return self

    def rotate(self, angle:float, axis:VectorType, origin:Optional[PointType]=None) -> 'TransformableBase':
        """Rotate by 'angle' around 'axis' going through 'origin';
        returns the same instance to enable chaining of transformations."""
        for point in self.points:
            point.rotate(angle, axis, origin)

        if self.edges is not None:
            for edge in self.edges:
                edge.rotate(angle, axis, origin)

        return self

    def scale(self, ratio:float, origin:Optional[PointType]=None) -> 'TransformableBase':
        """Scale with respect to given origin; returns the same instance
        to enable chaining of transformations."""
        for point in self.points:
            point.scale(ratio, origin)

        if self.edges is not None:
            for edge in self.edges:
                edge.scale(ratio, origin)

        return self
