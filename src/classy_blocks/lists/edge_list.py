from typing import List

from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.factory import factory

from classy_blocks.util import constants

class EdgeList:
    """Handling of the 'edges' part of blockMeshDict"""
    @property
    def edges(self) -> List[Edge]:
        """A convenient access to factory.registry"""
        return factory.registry

    @property
    def description(self) -> str:
        """Outputs a list of edges to be inserted into blockMeshDict"""
        out = "edges\n(\n"

        for edge in self.edges:
            out += f"\t{edge.description}\n"

        out += ");\n\n"

        return out
