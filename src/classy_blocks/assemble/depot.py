from typing import List, Union

from classy_blocks.construct.assemblies.assembly import Assembly
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import Shape
from classy_blocks.construct.stack import Stack

AdditiveType = Union[Operation, Shape, Stack, Assembly]


class Depot:
    """Collects, stores and serves user-added AdditiveType stuff"""

    def __init__(self) -> None:
        self.solids: List[AdditiveType] = []
        self.deleted: List[Operation] = []

    def add_solid(self, solid: AdditiveType) -> None:
        self.solids.append(solid)

    def delete_solid(self, operation: Operation) -> None:
        self.deleted.append(operation)

    @property
    def operations(self) -> List[Operation]:
        operations: List[Operation] = []

        for solid in self.solids:
            if isinstance(solid, Operation):
                operations.append(solid)
            else:
                operations += solid.operations

        return operations
