import dataclasses

from classy_blocks.construct import edges
from classy_blocks.items.edges.edge import Edge


@dataclasses.dataclass
class ProjectEdge(Edge):
    """Edge, projected to a specified geometry"""

    data: edges.Project

    @property
    def length(self):
        # can't say much about that length, eh?
        return super().length

    @property
    def description(self):
        return f"\tproject {self.vertex_1.index} {self.vertex_2.index} ({' '.join(self.data.label)})"
