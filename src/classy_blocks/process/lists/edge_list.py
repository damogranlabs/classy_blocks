from typing import List

from classy_blocks.process.items.vertex import Vertex
from classy_blocks.process.lists.vertex_list import VertexList
from classy_blocks.process.lists.block_list import BlockList

from classy_blocks.process.items.edge_ops import EdgeOps, factory

class EdgeList:
    """Handling of the 'edges' part of blockMeshDict"""
    def __init__(self):
        self.ops:List[EdgeOps] = []

    def find(self, vertex_1:Vertex, vertex_2:Vertex) -> EdgeOps:
        """checks if an edge with the same pair of vertices
        exists in self.edges already"""
        for edge in self.ops:
            mesh_set = set([vertex_1.index, vertex_2.index])
            edge_set = set([edge.vertex_1.index, edge.vertex_2.index])
            if mesh_set == edge_set:
                return edge

        raise RuntimeError(f"Edge not found: {str(vertex_1)}, {str(vertex_2)}")

    def collect(self, block_list:BlockList, vertex_list:VertexList) -> None:
        """Collect edges from this block;
        check for duplicates (same vertex pairs) and
        validity (no lines or straight-line arcs);
        remove edges that don't pass those tests"""
        for block in block_list.blocks:
            for edge_data in block.edges:
                point_1 = block.points[edge_data.index_1]
                vertex_1 = vertex_list.find(point_1)

                point_2 = block.points[edge_data.index_2]
                vertex_2 = vertex_list.find(point_2)

                try:
                    self.find(vertex_1, vertex_2)
                except RuntimeError:
                    # this edge doesn't exist yet;
                    # generate a new one and add it to the list
                    args = [
                        vertex_1,
                        vertex_2,
                        edge_data.kind
                    ] + edge_data.args

                    edge = factory.create(*args)

                    if edge.is_valid:
                        self.ops.append(edge)

    def output(self) -> str:
        """Outputs a list of edges to be inserted into blockMeshDict"""
        # s = "edges\n(\n"

        # for edge in self.edges:
        #     if edge.kind == "line":
        #         continue

        #     if edge.kind == "project":
        #         point_list =  f"({edge.points})"
        #     elif edge.kind == "arc":
        #         point_list = constants.vector_format(edge.points)
        #     else:
        #         point_list = "(" + " ".join([constants.vector_format(p) for p in edge.points]) + ")"


        #     s += f"\t{edge.kind} {edge.vertex_1.mesh_index} {edge.vertex_2.mesh_index} {point_list}\n"

        # s += ");\n\n"

        # return s
