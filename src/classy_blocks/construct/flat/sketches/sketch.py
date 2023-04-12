import abc
import copy

from typing import List, TypeVar, Optional

from classy_blocks.types import VectorType, PointType, NPPointType, NPVectorType
from classy_blocks.base.element import ElementBase
from classy_blocks.base import transforms as tr
from classy_blocks.construct.flat.face import Face

SketchT = TypeVar("SketchT", bound="Sketch")


class Sketch(ElementBase):
    """A collection of Faces that form the basis of a 3D Shape."""

    @property
    @abc.abstractmethod
    def faces(self) -> List[Face]:
        """Faces that form this sketch"""

    def copy(self: SketchT) -> SketchT:
        """Returns a copy of this sketch"""
        return copy.deepcopy(self)

    def translate(self: SketchT, displacement: VectorType) -> SketchT:
        for face in self.faces:
            face.translate(displacement)

        return self

    def rotate(self: SketchT, angle: float, axis: VectorType, origin: Optional[PointType] = None) -> SketchT:
        if origin is None:
            origin = self.center

        for face in self.faces:
            face.rotate(angle, axis, origin)

        return self

    def scale(self: SketchT, ratio: float, origin: Optional[PointType] = None) -> SketchT:
        if origin is None:
            origin = self.center

        for face in self.faces:
            face.scale(ratio, origin)

        return self

    def transform(self: SketchT, transform: List[tr.TransformationBase]) -> SketchT:
        """A function that transforms sketch_1 to sketch_2;
        a Loft will be made from those"""
        for t7m in transform:
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

    @property
    @abc.abstractmethod
    def center(self) -> NPPointType:
        """Center of this sketch"""

    @property
    def normal(self) -> NPVectorType:
        """Normal of this sketch"""
        return self.faces[0].normal

    @property
    @abc.abstractmethod
    def n_segments(self) -> int:
        """Number of outer faces"""
