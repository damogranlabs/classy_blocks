from typing import List, Optional

from classy_blocks.define.block import Block
from classy_blocks.define.primitives import Vertex, Edge, WrongEdgeTypeException

from classy_blocks.util import constants

class EdgeList:
    def __init__(self):
        self.edges:List[Edge] = []
    
    def find(self, vertex_1, vertex_2) -> Optional[Edge]:
        """checks if an edge with the same pair of vertices
        exists in self.edges already"""
        for e in self.edges:
            mesh_set = set([vertex_1.mesh_index, vertex_2.mesh_index])
            edge_set = set([e.vertex_1.mesh_index, e.vertex_2.mesh_index])
            if mesh_set == edge_set:
                return e

        return None

    def collect(self, blocks:List[Block]) -> None:
        """Collects all edges from all blocks,
        checks for duplicates (same vertex pairs) and
        checks for validity (no lines or straight-line arcs);
        removes edges that don't pass those tests"""
        for block in blocks:
            legit_edges = []

            for i, block_edge in enumerate(block.edges):
                # block.vertices by now contain index to mesh.vertices;
                # edge vertex therefore refers to actual mesh vertex
                v_1 = block.vertices[block_edge.block_index_1]
                v_2 = block.vertices[block_edge.block_index_2]

                block.edges[i].vertex_1 = v_1
                block.edges[i].vertex_2 = v_2

                if block_edge.type == "line":
                    continue

                if block_edge.is_valid:
                    if self.find(v_1, v_2) is None:
                        legit_edges.append(block_edge)

            self.edges += legit_edges
            # TODO: don't mess with the original block data
            block.edges = legit_edges

    def output(self) -> str:
        """Outputs a list of edges to be inserted into blockMeshDict"""
        s = "edges\n(\n"

        for edge in self.edges:
            if edge.type == "line":
                continue
            elif edge.type == "project":
                point_list =  f"({edge.points})"
            elif edge.type == "arc":
                point_list = constants.vector_format(edge.points)
            elif edge.type == "spline":
                point_list = "(" + " ".join([constants.vector_format(p) for p in edge.points]) + ")"
            else:
                raise WrongEdgeTypeException(edge.type)

            s += f"\t{edge.type} {edge.vertex_1.mesh_index} {edge.vertex_2.mesh_index} {point_list}\n"

        s += ");\n\n"

        return s