import abc
from collections.abc import Sequence

import numpy as np

from classy_blocks.base.element import ElementBase
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import Shape


class Assembly(ElementBase, abc.ABC):
    def __init__(self, shapes: Sequence[Shape]):
        self.shapes = shapes

    @property
    def parts(self):
        return self.shapes

    @property
    def center(self):
        return np.average([shape.center for shape in self.shapes])

    @property
    def operations(self) -> list[Operation]:
        operations: list[Operation] = []

        for shape in self.shapes:
            operations += shape.operations

        return operations
