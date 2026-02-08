import dataclasses

from classy_blocks.construct import edges
from classy_blocks.items.edges.edge import Edge


@dataclasses.dataclass
class CollapsedEdge(Edge):
    """A collapsed edge; only used to specify a three-sided pyramid"""

    data: edges.Collapsed

    @property
    def length(self):
        return 0

    @property
    def description(self):
        # no need to output that
        return ""

    @property
    def is_valid(self) -> bool:
        return True
