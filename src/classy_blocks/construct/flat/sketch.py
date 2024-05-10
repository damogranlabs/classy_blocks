import abc
import copy
from typing import ClassVar, List, Tuple, TypeVar

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.quad import Quad, smooth
from classy_blocks.types import NPPointType, NPVectorType, PointListType
from classy_blocks.util.constants import DTYPE

SketchT = TypeVar("SketchT", bound="Sketch")


class Sketch(ElementBase):
    """A collection of Faces that form the basis of a 3D Shape."""

    # indexes of faces that are to be chopped (within a Shape)
    # for axis 0 and axis 1; axis 2 is the 3rd dimension
    chops: ClassVar[List[List[int]]] = []

    @property
    @abc.abstractmethod
    def faces(self) -> List[Face]:
        """Faces that form this sketch"""

    @property
    @abc.abstractmethod
    def grid(self) -> List[List[Face]]:
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


class MappedSketch(Sketch):
    """A sketch that is created from predefined points.
    The points are connected to form quads which define Faces."""

    def __init__(self, positions: PointListType, quads: List[Tuple[int, int, int, int]], smooth_iter: int = 0):
        positions = np.array(positions, dtype=DTYPE)

        if smooth_iter > 0:
            positions = smooth(positions, quads, smooth_iter)

        self.quads = [Quad(positions, quad) for quad in quads]

        self.add_edges()

    def add_edges(self) -> None:
        """An optional method that will add edges to wherever they need to be."""

    @property
    def faces(self):
        """A 'flattened' grid"""
        return [quad.face for quad in self.quads]

    @property
    def grid(self):
        """Use a single-tier grid by default; override the method for more sophistication."""
        return [self.faces]

    @property
    def center(self) -> NPPointType:
        """Center of this sketch"""
        return np.average([face.center for face in self.faces], axis=0)
