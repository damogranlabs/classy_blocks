import abc
import copy

from typing import List, Optional

from classy_blocks.types import VectorType, PointType
from classy_blocks.base.transformable import TransformableBase
from classy_blocks.construct.flat.face import Face

class Sketch(TransformableBase, abc.ABC):
    """A collection of Faces that form the basis of a 3D Shape."""
    @property
    @abc.abstractmethod
    def faces(self) -> List[Face]:
        """Faces that form this sketch"""

    @abc.abstractmethod
    def copy(self) -> 'Sketch':
        """Returns a copy of this sketch"""
        return copy.deepcopy(self)

    def translate(self, displacement: VectorType) -> 'Sketch':
        for face in self.faces:
            face.translate(displacement)

        return self

    def rotate(self, angle: float, axis: VectorType, origin: Optional[PointType] = None) -> 'Sketch':
        for face in self.faces:
            face.rotate(angle, axis, origin)

        return self

    def scale(self, ratio: float, origin: Optional[PointType] = None) -> 'Scale':
        for face in self.faces:
            face.scale(ratio, origin)

        return self
