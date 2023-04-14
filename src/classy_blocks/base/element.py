import abc
import copy
from typing import TypeVar, List, Optional

from classy_blocks.types import PointType, VectorType, NPPointType
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

    def rotate(self: ElementBaseT, angle: float, axis: VectorType, origin: Optional[PointType] = None) -> ElementBaseT:
        """Rotate by 'angle' around 'axis' going through 'origin';
        returns the same instance to enable chaining of transformations."""
        if origin is None:
            origin = self.center

        for component in self.parts:
            component.rotate(angle, axis, origin)

        return self

    def scale(self: ElementBaseT, ratio: float, origin: Optional[PointType] = None) -> ElementBaseT:
        """Scale with respect to given origin; returns the same instance
        to enable chaining of transformations. If no origin is given,
        the entity is scaled with respect to its center"""
        if origin is None:
            origin = self.center

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

    @property
    @abc.abstractmethod
    def center(self) -> NPPointType:
        """Center of this entity; used as default origin for transforms"""

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
