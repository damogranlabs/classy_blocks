from typing import List

from classy_blocks.util.constants import EDGE_PAIRS

from classy_blocks.data.block import BlockData
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.factory import factory

class EdgeList:
    """Handling of the 'edges' part of blockMeshDict"""
    def __init__(self):
        self.edges:List[Edge] = []

    def find(self, vertex_1:Vertex, vertex_2:Vertex) -> Edge:
        """checks if an edge with the same pair of vertices
        exists in self.edges already"""
        for edge in self.edges:
            mesh_set = set([vertex_1.index, vertex_2.index])
            edge_set = set([edge.vertex_1.index, edge.vertex_2.index])
            if mesh_set == edge_set:
                return edge

        raise RuntimeError(f"Edge not found: {str(vertex_1)}, {str(vertex_2)}")

    def add(self, block_data:BlockData, vertices:List[Vertex]) -> List[Edge]:
        """Collect edges from this block;
        check for duplicates (same vertex pairs) and
        validity (no lines or straight-line arcs);
        remove edges that don't pass those tests"""
        edges = []

        for pair in EDGE_PAIRS:
            vertex_1 = vertices[pair[0]]
            vertex_2 = vertices[pair[1]]

            try:
                # if this edge exists in the list, return it regardless of what's
                # in block_data; re-definitions of the same edges are ignored
                edges.append(self.find(vertex_1, vertex_2))
            except RuntimeError:
                # this edge doesn't exist yet;
                # see if there's a new definition available in block_data
                try:
                    edge_data = block_data.get_edge(pair[0], pair[1])

                    args = [
                        vertices[edge_data.index_1],
                        vertices[edge_data.index_2],
                        edge_data.kind
                    ] + edge_data.args

                    edge = factory.create(*args)

                    if edge.is_valid:
                        self.edges.append(edge)
                        edges.append(edge)

                except RuntimeError:
                    # no new edges in the block definition
                    continue

        return edges

    @property
    def description(self) -> str:
        """Outputs a list of edges to be inserted into blockMeshDict"""
        out = "edges\n(\n"

        for edge in self.edges:
            out += f"\t{edge.description}\n"

        out += ");\n\n"

        return out
