from typing import ClassVar, List, Optional, Sequence, Union

import numpy as np

from classy_blocks.base import transforms as tr
from classy_blocks.base.element import ElementBase
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import LoftedShape
from classy_blocks.types import AxisType, PointType, VectorType
from classy_blocks.util import functions as f


class Stack(ElementBase):
    """A collection of topologically similar Shapes,
    stacked on top of each other."""

    shapes: ClassVar[List[LoftedShape]] = []

    @property
    def grid(self) -> List[List[List[Operation]]]:
        """Returns all operations as a 3-dimensional list;
        first two dimensions within a shape, the third along the stack."""
        # TODO: convert this to typed numpy array to support all indexing,
        # slicing and iterating, available on numpy arrays
        # Currently, the problem is typing support (Numpy arrays of exact object)

        # Invert the list so that z-value is the last
        shapes = np.asarray([shape.grid for shape in self.shapes], dtype="object")

        return np.swapaxes(shapes, 0, 2).tolist()

    def get_slice(self, axis: AxisType, index: int) -> List[Operation]:
        """Returns all operations along given axis;
        2 - same as shapes[index]
        0, 1 - like shape[axis][index] for each shape"""
        if axis == 0:
            return np.asarray(self.grid, dtype="object")[index, :, :].ravel().tolist()

        if axis == 1:
            return np.asarray(self.grid, dtype="object")[:, index, :].ravel().tolist()

        return self.shapes[index].operations

    def chop(self, **kwargs) -> None:
        """Adds a chop in lofted/extruded/revolved direction to one operation
        in each shape in the stack."""
        for shape in self.shapes:
            shape.grid[0][0].chop(2, **kwargs)

    @property
    def operations(self) -> List[Operation]:
        return f.flatten_2d_list([shape.operations for shape in self.shapes])

    @property
    def parts(self):
        return self.operations

    @property
    def center(self):
        return np.average([op.center for op in self.operations], axis=0)


class TransformedStack(Stack):
    """A stack where each next tier's sketch is transformed according to a list
    of transformations, passed to constructor. Arc edges can be created by specifying
    a mid_transforms list. The transformations there refer to base sketch - its vertices
    will be used as arc points for all lofted edges."""

    def __init__(
        self,
        base: Sketch,
        end_transforms: Sequence[tr.Transformation],
        repeats: int,
        mid_transforms: Optional[Sequence[tr.Transformation]] = None,
    ):
        sketch_1 = base

        for _ in range(repeats):
            sketch_2 = sketch_1.copy().transform(end_transforms)

            if mid_transforms is not None:
                sketch_mid = sketch_1.copy().transform(mid_transforms)
            else:
                sketch_mid = None

            shape = LoftedShape(sketch_1, sketch_2, sketch_mid)

            self.shapes.append(shape)
            sketch_1 = sketch_2


class ExtrudedStack(TransformedStack):
    """Extruded shapes, stacked on top of each other.
    Amount is overall 'height' of the stack."""

    def __init__(self, base: Sketch, amount: Union[float, VectorType], repeats: int):
        if isinstance(amount, float) or isinstance(amount, int):
            extrude_vector = base.normal * amount / repeats
        else:
            extrude_vector = np.asarray(amount) / repeats

        super().__init__(base, [tr.Translation(extrude_vector)], repeats)


class RevolvedStack(TransformedStack):
    """Revolved shapes, stacked around the given center.
    Angle given is overall and is divided by repeats for each tier."""

    def __init__(self, base: Sketch, angle: float, axis: VectorType, origin: PointType, repeats: int):
        super().__init__(
            base,
            [tr.Rotation(axis, angle / repeats, origin)],
            repeats,
            [tr.Rotation(axis, angle / repeats / 2, origin)],
        )
