import abc
import copy
from typing import ClassVar, TypeVar

from classy_blocks.base.element import ElementBase
from classy_blocks.cbtyping import NPPointType, NPVectorType
from classy_blocks.construct.flat.face import Face

SketchT = TypeVar("SketchT", bound="Sketch")


class Sketch(ElementBase):
    """A collection of Faces that form the basis of a 3D Shape."""

    # indexes of faces that are to be chopped (within a Shape)
    # for axis 0 and axis 1; axis 2 is the 3rd dimension
    chops: ClassVar[list[list[int]]] = []

    @property
    @abc.abstractmethod
    def faces(self) -> list[Face]:
        """Faces that form this sketch"""

    @property
    @abc.abstractmethod
    def grid(self) -> list[list[Face]]:
        """A 2-dimensional list of faces that form this sketch;
        addressed as x-y for cartesian sketches and as radius-angle
        for radial sketches.

        For instance, a 2x3 cartesian grid will obviously contain 2 lists,
        each of 3 faces but a disk (cylinder) grid
        will contain 2 lists, first with 4 core faces and the other with 8 outer faces.
        A simple Annulus sketch will only contain one list with all the faces."""

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
    def parts(self):
        return self.faces
