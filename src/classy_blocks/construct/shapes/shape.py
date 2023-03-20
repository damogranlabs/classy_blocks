import abc

from typing import TypeVar, Optional

import numpy as np

from classy_blocks.types import VectorType, PointType, NPPointType
from classy_blocks.base.additive import AdditiveBase
from classy_blocks.base.transformable import TransformableBase

ShapeT = TypeVar('ShapeT', bound='Shape')

class Shape(AdditiveBase, TransformableBase, abc.ABC):
    """A collection of Operations that form a predefined
    parametric shape"""
    def set_cell_zone(self, cell_zone:str) -> None:
        """Sets cell zone for all blocks in this shape"""
        for op in self.operations:
            op.set_cell_zone(cell_zone)

    def translate(self: ShapeT, displacement: VectorType) -> ShapeT:
        for operation in self.operations:
            operation.translate(displacement)

        return self

    def rotate(self: ShapeT, angle: float, axis: VectorType, origin: Optional[PointType]=None) -> ShapeT:
        if origin is None:
            origin = self.center

        for operation in self.operations:
            operation.rotate(angle, axis, origin)

        return self

    def scale(self: ShapeT, ratio: float, origin:Optional[PointType]=None) -> ShapeT:
        if origin is None:
            origin = self.center

        for operation in self.operations:
            operation.scale(ratio, origin)

        return self

    @property
    def center(self) -> NPPointType:
        """Geometric mean of centers of all operations"""
        return np.average([operation.center for operation in self.operations], axis=0)
