import dataclasses

from typing import Callable, ClassVar

import numpy as np

from classy_blocks.items.edges.arcs.arc_base import ArcEdgeBase
from classy_blocks.types import PointType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

@dataclasses.dataclass
class ArcEdge(ArcEdgeBase):
    """Arc edge: defined by a single point"""
    kind: ClassVar[str] = "arc"
    arc_point: PointType

    @property
    def third_point(self):
        return self.arc_point

    def transform(self, function: Callable):
        self.arc_point = function(self.arc_point)

        return self

    @property
    def is_valid(self):
        if super().is_valid:
            # TODO: TEST
            # if case vertex1, vertex2 and point in between
            # are collinear, blockMesh will find an arc with
            # infinite radius and crash.
            # so, check for collinearity; if the three points
            # are actually collinear, this edge is redundant and can be
            # silently dropped

            # cross-product of three collinear vertices must be zero
            arm_1 = self.vertex_1.pos - self.third_point
            arm_2 = self.vertex_2.pos - self.third_point

            return abs(f.norm(np.cross(arm_1, arm_2))) > constants.tol

        return False
