import dataclasses

from typing import ClassVar

from classy_blocks.items.edges.edge import Edge

@dataclasses.dataclass
class LineEdge(Edge):
    """A default Line edge; doesn't need an explicit definition and is not output to blockMeshDict"""

    kind: ClassVar[str] = "line"

    @property
    def length(self):
        # straight line
        return super().length
    
    @property
    def description(self):
        # no need to output that
        return ''