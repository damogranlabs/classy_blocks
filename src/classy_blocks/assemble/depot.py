from typing import Union

from classy_blocks.construct.assemblies.assembly import Assembly
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import Shape
from classy_blocks.construct.stack import Stack

AdditiveType = Union[Operation, Shape, Stack, Assembly]


class Depot:
    """Collects, stores and serves user-added AdditiveType stuff"""

    def __init__(self) -> None:
        self.solids: list[AdditiveType] = []
        self.deleted: list[Operation] = []

    def add_solid(self, solid: AdditiveType) -> None:
        self.solids.append(solid)

    def delete_solid(self, operation: Operation) -> None:
        self.deleted.append(operation)

    @property
    def operations(self) -> list[Operation]:
        operations: list[Operation] = []

        for solid in self.solids:
            if isinstance(solid, Operation):
                operations.append(solid)
            else:
                operations += solid.operations

        return operations
