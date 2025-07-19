from collections.abc import Sequence
from typing import Optional, Union

import numpy as np

from classy_blocks.base import transforms as tr
from classy_blocks.base.element import ElementBase
from classy_blocks.cbtyping import DirectionType, PointType, VectorType
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import LoftedShape
from classy_blocks.util import functions as f


class Stack(ElementBase):
    """A collection of topologically similar Shapes,
    stacked on top of each other."""

    def __init__(self) -> None:
        self.shapes: list[LoftedShape] = []

    @property
    def grid(self):
        """Returns all operations as a 3-dimensional list;
        first two dimensions within a shape, the third along the stack."""
        return [shape.grid for shape in self.shapes]

    def get_slice(self, axis: DirectionType, index: int) -> list[Operation]:
        """Returns all operation with given index in specified axis.
        For cartesian grids this is equivalent to 'lofts on the same plane';
        This does not work with custom/mapped sketches that do not
        conform to a cartesian grid.

        Example:
        A stack that consists of 3 shapes, created from a 2x5 grid.
        - get_slice(0, i) will return 15 operations (5x3, all operations with the same x-coordinate),
        - get_slice(1, i) will return 6 operations (2x3, all with the same y-coordinate),
        - get_slice(2, i) will return 10 operations (2x5, all with the same z-coordinate)."""

        if axis == 2:
            return self.shapes[index].operations

        operations: list[Operation] = []

        if axis == 0:
            for shape in self.shapes:
                operations += [shape.grid[x][index] for x in range(len(shape.grid))]
        else:
            for shape in self.shapes:
                operations += [shape.grid[index][y] for y in range(len(shape.grid[index]))]

        return operations

    def chop(self, **kwargs) -> None:
        """Adds a chop in lofted/extruded/revolved direction to one operation
        in each shape in the stack."""
        for shape in self.shapes:
            shape.grid[0][0].chop(2, **kwargs)

    @property
    def operations(self) -> list[Operation]:
        return f.flatten_2d_list([shape.operations for shape in self.shapes])

    @property
    def parts(self):
        return self.shapes

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
        super().__init__()

        sketch_1 = base

        for _ in range(repeats):
            sketch_2 = sketch_1.copy().transform(end_transforms)

            if mid_transforms is not None:
                sketch_mid = sketch_1.copy().transform(mid_transforms)
            else:
                sketch_mid = None

            shape = LoftedShape(sketch_1, sketch_2, sketch_mid)

            self.shapes.append(shape)
            sketch_1 = sketch_2.copy()


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
