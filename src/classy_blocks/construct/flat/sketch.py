import abc
import copy

from typing import List, Optional, TypeVar

import numpy as np

from classy_blocks.types import VectorType, PointType, NPPointType
from classy_blocks.base.transformable import TransformableBase
from classy_blocks.construct.flat.face import Face

SketchT = TypeVar('SketchT', bound='Sketch')

class Sketch(TransformableBase, abc.ABC):
    """A collection of Faces that form the basis of a 3D Shape."""
    @property
    @abc.abstractmethod
    def faces(self) -> List[Face]:
        """Faces that form this sketch"""

    def copy(self:SketchT) -> SketchT:
        """Returns a copy of this sketch"""
        return copy.deepcopy(self)

    def translate(self:SketchT, displacement: VectorType) -> SketchT:
        for face in self.faces:
            face.translate(displacement)

        return self

    def rotate(self:SketchT, angle: float, axis: VectorType, origin: Optional[PointType] = None) -> SketchT:
        for face in self.faces:
            face.rotate(angle, axis, origin)

        return self

    def scale(self:SketchT, ratio: float, origin: Optional[PointType] = None) -> SketchT:
        if origin is None:
            origin = self.center

        for face in self.faces:
            face.scale(ratio, origin)

        return self

    @property
    def center(self) -> NPPointType:
        """Center of this sketch; an average of all faces"""
        return np.average([f.center for f in self.faces], axis=0)

    @property
    @abc.abstractmethod
    def n_segments(self) -> int:
        """Number of faces defining this annulus"""
