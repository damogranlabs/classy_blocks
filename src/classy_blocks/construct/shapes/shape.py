"""Abstract base classes for different Shape types"""

import abc
from typing import Generic, List, Optional, Sequence, TypeVar

import numpy as np

from classy_blocks.base import transforms as tr
from classy_blocks.base.element import ElementBase
from classy_blocks.construct.edges import Arc
from classy_blocks.construct.flat.sketches.sketch import SketchT
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.types import AxisType, NPPointType, OrientType

ShapeT = TypeVar("ShapeT", bound="Shape")


class Shape(ElementBase):
    """A collection of Operations that form a predefined
    parametric shape"""

    @property
    @abc.abstractmethod
    def operations(self) -> List[Operation]:
        """Operations from which the shape is build"""

    @property
    def parts(self):
        return self.operations

    def set_cell_zone(self, cell_zone: str) -> None:
        """Sets cell zone for all blocks in this shape"""
        for operation in self.operations:
            operation.set_cell_zone(cell_zone)

    @property
    def center(self) -> NPPointType:
        """Geometric mean of centers of all operations"""
        return np.average([operation.center for operation in self.operations], axis=0)


class SketchedShape(Shape, Generic[SketchT]):
    """A Shape, obtained by taking a Sketch and transforming it once
    or twice (middle/end cross-section), then making profiled Lofts
    from calculated cross-sections (Elbow, Cylinder, Ring, ..."""

    axial_axis: AxisType = 2  # Axis along which 'outer sides' run
    radial_axis: AxisType = 0  # Axis that goes from center to 'outer side'
    tangential_axis: AxisType = 1  # Axis that goes around the circumference of the shape

    start_patch: OrientType = "bottom"  # Sides of blocks that define the start patch
    end_patch: OrientType = "top"  # Sides of blocks that define the end patch"""
    outer_patch: OrientType = "right"  # Sides of blocks that define the outer surface

    def __init__(
        self,
        sketch_1: SketchT,
        sketch_2_transform: Sequence[tr.Transformation],
        sketch_mid_transform: Optional[Sequence[tr.Transformation]] = None,
    ):
        # start with sketch_1 and transform it
        # using the _transform function(transform_2_args) to obtain sketch_2;
        # use _transform function(transform_mid_args) to obtain mid sketch
        # (only if applicable)
        self.sketch_1 = sketch_1
        self.sketch_2 = sketch_1.copy().transform(sketch_2_transform)

        self.sketch_mid: Optional[SketchT] = None
        if sketch_mid_transform is not None:
            self.sketch_mid = sketch_1.copy().transform(sketch_mid_transform)

        self.lofts: List[Loft] = []

        for i, face_1 in enumerate(self.sketch_1.faces):
            face_2 = self.sketch_2.faces[i]

            loft = Loft(face_1, face_2)

            # add edges, if applicable
            if self.sketch_mid:
                face_mid = self.sketch_mid.faces[i]

                for i, point in enumerate(face_mid.points):
                    loft.add_side_edge(i, Arc(point.position))

            self.lofts.append(loft)

    def chop_axial(self, **kwargs):
        """Chop the shape between start and end face"""
        self.operations[0].chop(self.axial_axis, **kwargs)

    @abc.abstractmethod
    def chop_radial(self, **kwargs):
        """Chop in the radial direction"""

    @abc.abstractmethod
    def chop_tangential(self, **kwargs):
        """Circumferential chop"""

    def set_start_patch(self, name: str) -> None:
        """Assign the faces of start sketch to a named patch"""
        for operation in self.operations:
            operation.set_patch(self.start_patch, name)

    def set_end_patch(self, name: str) -> None:
        """Assign the faces of end sketch to a named patch"""
        for operation in self.operations:
            operation.set_patch(self.end_patch, name)

    @property
    def operations(self):
        return self.lofts

    @property
    @abc.abstractmethod
    def shell(self) -> List[Loft]:
        """Operations on the outside of the shape"""
