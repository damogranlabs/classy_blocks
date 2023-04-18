import dataclasses

from classy_blocks.construct import edges
from classy_blocks.items.edges.arcs.arc_base import ArcEdgeBase


@dataclasses.dataclass
class ArcEdge(ArcEdgeBase):
    """Arc edge: defined by a single point"""

    data: edges.Arc

    @property
    def third_point(self):
        return self.data.point
