import dataclasses
import abc

import numpy as np

from classy_blocks.items.edges.edge import Edge
from classy_blocks.types import PointType
from classy_blocks.util import constants
from classy_blocks.util import functions as f

@dataclasses.dataclass
class ArcEdgeBase(Edge, abc.ABC):
    """Base for all arc-based edges (arc, origin, angle)"""
    @property
    @abc.abstractmethod
    def third_point(self) -> PointType:
        """The third point that defines the arc, regardless of how it was specified"""

    @property
    def length(self) -> float:
        try:
            return f.arc_length_3point(self.vertex_1.pos, self.third_point, self.vertex_2.pos)
        except ValueError:
            return super().length

    @property
    def description(self):
        # it's always 'arc' for arc edges
        return f"arc {self.vertex_1.index} {self.vertex_2.index} " + \
            constants.vector_format(self.third_point)

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

            return abs(f.norm(np.cross(arm_1, arm_2))) > constants.TOL

        return False
