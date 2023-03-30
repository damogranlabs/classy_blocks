import dataclasses

from classy_blocks.construct import edges
from classy_blocks.items.edges.edge import Edge


@dataclasses.dataclass
class LineEdge(Edge):
    """A default Line edge; doesn't need an explicit definition and is not output to blockMeshDict"""

    data: edges.Line

    @property
    def length(self):
        # straight line
        return super().length

    @property
    def description(self):
        # no need to output that
        return ""
