"""Dataclasses for packing combinations of transforms of <anything>
into an easily digestable function/method arguments"""

import dataclasses
from typing import Optional

from classy_blocks.types import PointType, VectorType


@dataclasses.dataclass
class Transformation:
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
    origin: Optional[PointType] = None


@dataclasses.dataclass
class Scaling(Transformation):
    """Parameters required to scale an entity"""

    ratio: float
    origin: Optional[PointType] = None


@dataclasses.dataclass
class Mirror(Transformation):
    """Parameters required to mirror an entity around an
    arbitrary plane"""

    normal: VectorType
    origin: Optional[PointType] = None
