import abc
import copy
from typing import TypeVar, List

from classy_blocks.types import PointType, VectorType
from classy_blocks.base import transforms as tr

ElementBaseT = TypeVar("ElementBaseT", bound="ElementBase")


class ElementBase(abc.ABC):
    """Base class for mesh-building elements and tools
    for manipulation thereof."""

    def translate(self: ElementBaseT, displacement: VectorType) -> ElementBaseT:
        """Move by displacement vector; returns the same instance
        to enable chaining of transformations."""
        for component in self.parts:
            component.translate(displacement)

        return self

    def rotate(self: ElementBaseT, angle: float, axis: VectorType, origin: PointType) -> ElementBaseT:
        """Rotate by 'angle' around 'axis' going through 'origin';
        returns the same instance to enable chaining of transformations."""
        for component in self.parts:
            component.rotate(angle, axis, origin)

        return self

    def scale(self: ElementBaseT, ratio: float, origin: PointType) -> ElementBaseT:
        """Scale with respect to given origin; returns the same instance
        to enable chaining of transformations. If no origin is given,
        the entity is scaled with respect to its center"""
        for component in self.parts:
            component.scale(ratio, origin)

        return self

    def copy(self: ElementBaseT) -> ElementBaseT:
        """Returns a copy of this object"""
        return copy.deepcopy(self)

    @property
    @abc.abstractmethod
    def parts(self: ElementBaseT) -> List[ElementBaseT]:
        """A list of lower-dimension elements
        from which this element is built, for instance:
        - an edge has a single arc point,
        - a face has 4 points and 4 edges,
        - an Operation has 2 faces and 4 side edges"""

    def transform(self: ElementBaseT, transforms: List[tr.Transformation]) -> ElementBaseT:
        """A function that transforms  to sketch_2;
        a Loft will be made from those"""
        for t7m in transforms:
            for component in self.parts:
                if isinstance(t7m, tr.Translation):
                    component.translate(t7m.displacement)
                    continue

                if isinstance(t7m, tr.Rotation):
                    component.rotate(t7m.angle, t7m.axis, t7m.origin)
                    continue

                if isinstance(t7m, tr.Scaling):
                    component.scale(t7m.ratio, t7m.origin)
                    continue

        return self
