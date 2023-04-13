"""Dataclasses for packing combinations of transforms of <anything>
into an easily digestable function/method arguments"""
import abc
import dataclasses

from classy_blocks.types import PointType, VectorType


@dataclasses.dataclass
class Transformation(abc.ABC):
    """A superclass that addresses all
    dataclasses for transformation parameters"""


@dataclasses.dataclass
class Translation(Transformation):
    """Parameters required to translate an entity"""

    displacement: VectorType


@dataclasses.dataclass
class Rotation(Transformation):
    """Parameters required to rotate an entity"""

    axis: VectorType
    angle: float
    origin: PointType


@dataclasses.dataclass
class Scaling(Transformation):
    """Parameters required to scale an entity"""

    ratio: float
    origin: PointType
