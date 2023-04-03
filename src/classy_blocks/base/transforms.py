"""Dataclasses for packing combinations of transforms of <anything>
into an easily digestable function/method arguments"""
from typing import List, Union, Optional

import dataclasses

from classy_blocks.types import PointType, VectorType


@dataclasses.dataclass
class Translation:
    """Parameters required to translate an entity"""

    displacement: VectorType


@dataclasses.dataclass
class Rotation:
    """Parameters required to rotate an entity"""

    axis: VectorType
    angle: float
    origin: Optional[PointType] = None


@dataclasses.dataclass
class Scaling:
    """Parameters required to scale an entity"""

    ratio: float
    origin: Optional[PointType] = None


@dataclasses.dataclass
class Transformation:
    """A combo of all supported transforms that can be imposed on an entity"""

    transforms: List[Union[Translation, Rotation, Scaling]]
