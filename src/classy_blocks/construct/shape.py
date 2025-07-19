"""Abstract base classes for different Shape types"""

import abc
from typing import Generic, Optional, TypeVar, Union

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.base.exceptions import ShapeCreationError
from classy_blocks.cbtyping import DirectionType, NPPointType, VectorType
from classy_blocks.construct.edges import Angle
from classy_blocks.construct.flat.sketch import Sketch, SketchT
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.util import functions as f

ShapeT = TypeVar("ShapeT", bound="Shape")


class Shape(ElementBase, abc.ABC):
    """A collection of Operations that form a predefined
    parametric shape"""

    @property
    @abc.abstractmethod
    def operations(self) -> list[Operation]:
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

    @property
    @abc.abstractmethod
    def grid(self) -> list[list[Operation]]:
        """A 2-dimensional array consisting of Operations, grouped by their position
        (like [[core], [shell]] or [[rows], [columns]])"""


class LoftedShape(Shape, abc.ABC, Generic[SketchT]):
    """A Shape, obtained by taking a two and transforming it once
    or twice (middle/end cross-section), then making profiled Lofts
    from calculated cross-sections (Elbow, Cylinder, Ring, ..."""

    def __init__(
        self, sketch_1: SketchT, sketch_2: SketchT, sketch_mid: Optional[Union[SketchT, list[SketchT]]] = None
    ):
        if len(sketch_1.faces) != len(sketch_2.faces):
            raise ShapeCreationError("Start and end sketch have different number of faces!")

        if sketch_mid is None:
            sketch_mid = []
        else:
            if not isinstance(sketch_mid, list):
                sketch_mid = [sketch_mid]

            if any([len(sketch_mid_i.faces) != len(sketch_1.faces) for sketch_mid_i in sketch_mid]):
                raise ShapeCreationError("Mid sketch has a different number of faces from start/end!")

        self.sketch_1 = sketch_1
        self.sketch_2 = sketch_2
        self.sketch_mid = sketch_mid

        self.lofts: list[list[Loft]] = []

        for i, list_1 in enumerate(self.sketch_1.grid):
            self.lofts.append([])

            for j, face_1 in enumerate(list_1):
                face_2 = self.sketch_2.grid[i][j]

                mid_faces = [sketch.grid[i][j] for sketch in sketch_mid]
                loft = Loft.from_series([face_1, *mid_faces, face_2])

                self.lofts[-1].append(loft)

    def set_start_patch(self, name: str) -> None:
        """Assign the faces of start sketch to a named patch"""
        for operation in self.operations:
            operation.set_patch("bottom", name)

    def set_end_patch(self, name: str) -> None:
        """Assign the faces of end sketch to a named patch"""
        for operation in self.operations:
            operation.set_patch("top", name)

    @property
    def operations(self):
        return f.flatten_2d_list(self.lofts)

    @property
    def grid(self):
        """Analogous to Sketch's grid but corresponsing operations are returned"""
        return self.lofts

    def chop(self, axis: DirectionType, **kwargs) -> None:
        """Chops operations along given axis.
        Only axis 0 and 1 are allowed as defined in sketch_1"""
        if axis == 2:
            self.operations[0].chop(2, **kwargs)
        else:
            for index in self.sketch_1.chops[axis]:
                self.operations[index].chop(axis, **kwargs)


class ExtrudedShape(LoftedShape):
    """Analogous to an Extrude operation but on a Sketch"""

    # TODO: make operations universal - work on faces or sketches equally

    def __init__(self, sketch: Sketch, amount: Union[float, VectorType]):
        if isinstance(amount, float) or isinstance(amount, int):
            extrude_vector = sketch.normal * amount
        else:
            extrude_vector = np.asarray(amount)

        bottom_sketch = sketch
        top_sketch = bottom_sketch.copy().translate(extrude_vector)

        super().__init__(bottom_sketch, top_sketch)


class RevolvedShape(LoftedShape):
    def __init__(self, sketch: Sketch, angle: float, axis: VectorType, origin: VectorType):
        bottom_sketch = sketch
        top_sketch = bottom_sketch.copy().rotate(angle, axis, origin)

        super().__init__(bottom_sketch, top_sketch)

        # add side edges
        for operation in self.operations:
            for i in range(4):
                operation.add_side_edge(i, Angle(angle, axis))
