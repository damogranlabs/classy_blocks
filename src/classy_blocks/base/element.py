import abc
import copy
from typing import TypeVar, List

from classy_blocks.types import PointType, VectorType
from classy_blocks.base import transforms as tr

ElementBaseT = TypeVar("ElementBaseT", bound="ElementBase")


class ElementBase(abc.ABC):
    """Base class for mesh-building elements and tools
    for manipulation thereof."""

    @abc.abstractmethod
    def translate(self: ElementBaseT, displacement: VectorType) -> ElementBaseT:
        """Move by displacement vector; returns the same instance
        to enable chaining of transformations."""

    @abc.abstractmethod
    def rotate(self: ElementBaseT, angle: float, axis: VectorType, origin: PointType) -> ElementBaseT:
        """Rotate by 'angle' around 'axis' going through 'origin';
        returns the same instance to enable chaining of transformations."""

    @abc.abstractmethod
    def scale(self: ElementBaseT, ratio: float, origin: PointType) -> ElementBaseT:
        """Scale with respect to given origin; returns the same instance
        to enable chaining of transformations. If no origin is given,
        the entity is scaled with respect to its center"""

    def copy(self: ElementBaseT) -> ElementBaseT:
        """Returns a copy of this object"""
        return copy.deepcopy(self)

    @property
    @abc.abstractmethod
    def components(self: ElementBaseT) -> List[ElementBaseT]:
        """A list of lower-dimension elements
        from which this element is built, for instance:
        - an edge has a single arc point,
        - a face has 4 points and 4 edges,
        - an Operation has 2 faces and 4 side edges"""
        return []

    def transform(self: ElementBaseT, transforms: List[tr.TransformationBase]) -> ElementBaseT:
        """A function that transforms  to sketch_2;
        a Loft will be made from those"""
        for t7m in transforms:
            if isinstance(t7m, tr.Translation):
                self.translate(t7m.displacement)
                continue

            if isinstance(t7m, tr.Rotation):
                self.rotate(t7m.angle, t7m.axis, t7m.origin)
                continue

            if isinstance(t7m, tr.Scaling):
                self.scale(t7m.ratio, t7m.origin)
                continue

        return self
