import dataclasses

from typing import Callable, ClassVar, List

import numpy as np

from classy_blocks.items.edges.edge import Edge
from classy_blocks.types import PointType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

@dataclasses.dataclass
class ProjectEdge(Edge):
    """Edge, projected to a specified geometry"""

    kind: ClassVar[str] = "project"

    geometry: str # TODO: a list of 2 strings! to project to an intersection of 2 surfaces

    @property
    def args(self) -> List:
        return super().args

    @property
    def length(self):
        # can't say much about that length, eh?
        return super().length
    
    @property
    def description(self):
        return f"project {self.vertex_1.index} {self.vertex_2.index} ({self.geometry})"