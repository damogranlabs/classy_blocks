from typing import List

from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.operations.loft import Loft
from classy_blocks.construct.shape import LoftedShape, RevolvedShape
from classy_blocks.types import PointType, VectorType


class Stack:
    """A collection of topologically similar Shapes,
    stacked on top of each other."""

    shapes: List[LoftedShape] = []

    @property
    def grid(self) -> List[List[List[Loft]]]:
        """Returns a 3-dimensional list of operations;
        first two dimensions within a shape, the third along the stack"""
        return [shape.grid for shape in self.shapes]


class RevolvedStack(Stack):
    """Revolved shapes, stacked around the given center"""

    def __init__(self, base: Sketch, angle: float, axis: VectorType, origin: PointType, repeats: int):
        for _ in range(repeats):
            shape = RevolvedShape(base, angle, axis, origin)
            self.shapes.append(shape)
            base = shape.sketch_2
