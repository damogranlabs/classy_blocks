import abc
import copy
from typing import Dict, List, Optional, Sequence, TypeVar

from classy_blocks.base import transforms as tr
from classy_blocks.types import NPPointType, PointType, VectorType

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

    def mirror(self: ElementBaseT, normal: VectorType, origin: Optional[PointType] = None) -> ElementBaseT:
        """Mirror around a plane, defined by a normal vector and passing through origin;
        if origin is not given, [0, 0, 0] is assumed"""
        if origin is None:
            origin = [0, 0, 0]

        for component in self.parts:
            component.mirror(normal, origin)

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

    @property
    def geometry(self) -> Optional[Dict]:
        """A searchable surface, defined in an entity itself;
        (like, for instance, sphere's blocks are automatically
        projected to an ad-hoc defined searchableSphere"""
        return None

    def transform(self: ElementBaseT, transforms: Sequence[tr.Transformation]) -> ElementBaseT:
        """A function that transforms  to sketch_2;
        a Loft will be made from those"""

        for t7m in transforms:
            # remember center or it will change during transformation
            # of each self.part
            center = self.center

            for part in self.parts:
                if isinstance(t7m, tr.Translation):
                    part.translate(t7m.displacement)
                    continue

                if isinstance(t7m, tr.Rotation):
                    origin = t7m.origin
                    if origin is None:
                        origin = center

                    part.rotate(t7m.angle, t7m.axis, origin=origin)
                    continue

                if isinstance(t7m, tr.Scaling):
                    origin = t7m.origin
                    if origin is None:
                        origin = center

                    part.scale(t7m.ratio, origin=origin)
                    continue

                if isinstance(t7m, tr.Mirror):
                    origin = t7m.origin
                    if origin is None:
                        origin = [0, 0, 0]

                    part.mirror(t7m.normal, origin=origin)
                    continue

        return self
