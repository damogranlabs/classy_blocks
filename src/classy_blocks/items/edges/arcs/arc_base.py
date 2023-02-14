import dataclasses
import abc

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
        return f.arc_length_3point(self.vertex_1.pos, self.third_point, self.vertex_2.pos)

    @property
    def description(self):
        # it's always 'arc' for arc edges
        return f"arc {self.vertex_1.index} {self.vertex_2.index} " + \
            constants.vector_format(self.third_point)
