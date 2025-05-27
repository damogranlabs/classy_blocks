from typing import List, Union

from classy_blocks.construct.assemblies.assembly import Assembly
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import Shape
from classy_blocks.construct.stack import Stack

AdditiveType = Union[Operation, Shape, Stack, Assembly]


class SolidDepot:
    """Collects, stores and serves AdditiveType stuff"""

    def __init__(self):
        self.solids: List[AdditiveType] = []
        self.deleted: List[Operation] = []

    def add(self, solid: AdditiveType) -> None:
        self.solids.append(solid)

    def delete(self, operation: Operation) -> None:
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

    def get_geometry(self) -> List[dict]:
        geom = []

        for solid in self.solids:
            if solid.geometry is not None:
                geom.append(solid.geometry)

        return geom


# class PatchData
