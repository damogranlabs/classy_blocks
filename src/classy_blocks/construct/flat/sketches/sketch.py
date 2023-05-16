import abc
import copy
from typing import List, TypeVar

from classy_blocks.base.element import ElementBase
from classy_blocks.construct.flat.face import Face
from classy_blocks.types import NPPointType, NPVectorType

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

    @property
    def parts(self):
        return self.faces
