from typing import ClassVar, List, Union

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import ExtrudedShape, LoftedShape, RevolvedShape
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


class ExtrudedStack(Stack):
    """Extruded shapes, stacked on top of each other"""

    def __init__(self, base: Sketch, amount: Union[float, VectorType], repeats: int):
        if isinstance(amount, float) or isinstance(amount, int):
            extrude_vector = base.normal * amount
        else:
            extrude_vector = np.asarray(amount)

        for _ in range(repeats):
            shape = ExtrudedShape(base, extrude_vector / repeats)
            self.shapes.append(shape)
            base = shape.sketch_2


class RevolvedStack(Stack):
    """Revolved shapes, stacked around the given center"""

    def __init__(self, base: Sketch, angle: float, axis: VectorType, origin: PointType, repeats: int):
        for _ in range(repeats):
            shape = RevolvedShape(base, angle, axis, origin)
            self.shapes.append(shape)
            base = shape.sketch_2
