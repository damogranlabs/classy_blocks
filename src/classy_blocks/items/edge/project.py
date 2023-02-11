import dataclasses

from typing import Callable, ClassVar

import numpy as np

from classy_blocks.items.edge.base import Edge
from classy_blocks.types import PointType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

@dataclasses.dataclass
class ProjectEdge(Edge):
    """Edge, projected to a specified geometry"""

    kind: ClassVar[str] = "project"

    geometry: str