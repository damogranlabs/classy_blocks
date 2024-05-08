from typing import ClassVar, List

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import LoftedShape, RevolvedShape
from classy_blocks.types import PointType, VectorType
from classy_blocks.util import functions as f


class Stack(ElementBase):
    """A collection of topologically similar Shapes,
    stacked on top of each other."""

    shapes: ClassVar[List[LoftedShape]] = []

    @property
    def grid(self) -> List[List[List[Loft]]]:
        """Returns a 3-dimensional list of operations;
        first two dimensions within a shape, the third along the stack"""
        return [shape.grid for shape in self.shapes]

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


class RevolvedStack(Stack):
    """Revolved shapes, stacked around the given center"""

    def __init__(self, base: Sketch, angle: float, axis: VectorType, origin: PointType, repeats: int):
        for _ in range(repeats):
            shape = RevolvedShape(base, angle, axis, origin)
            self.shapes.append(shape)
            base = shape.sketch_2
