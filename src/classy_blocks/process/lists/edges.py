from typing import List, Optional

from classy_blocks.define.edge import Edge

from classy_blocks.define.point import Point
from classy_blocks.process.items.hexa import Hexa
from classy_blocks.process.items.edge import Edge, factory

from classy_blocks.util import constants

class EdgeList:
    def __init__(self):
        self.edges = []
    
    def find(self, vertex_1:Point, vertex_2:Point):
        """checks if an edge with the same pair of vertices
        exists in self.edges already"""
        for e in self.edges:
            mesh_set = set([vertex_1.index, vertex_2.index])
            edge_set = set([e.vertex_1.index, e.vertex_2.index])
            if mesh_set == edge_set:
                return e

        return None

    def add(self, vertex_1:Point, vertex_2:Point, curve:Edge):
        if self.find(vertex_1, vertex_2) is not None:
            # don't overwrite an existing edge
            return
        
        self.edges.append(factory.create(vertex_1, vertex_2, curve))

    def collect(self, hexas:List[Hexa]) -> None:
        """Collects all edges from all blocks,
        checks for duplicates (same vertex pairs) and
        checks for validity (no lines or straight-line arcs);
        removes edges that don't pass those tests"""
        for hexa in hexas:
            legit_edges = []

            for i, curve in enumerate(.curves):
                # block.vertices by now contain index to mesh.vertices;
                # edge vertex therefore refers to actual mesh vertex
                v_1 = block.vertices[block_edge.index_1]
                v_2 = block.vertices[block_edge.index_2]

                block.edges[i].vertex_1 = v_1
                block.edges[i].vertex_2 = v_2

                if block_edge.is_valid():
                    if self.find(v_1, v_2) is None:
                        legit_edges.append(block_edge)

            self.edges += legit_edges
            # TODO: don't mess with the original block data
            block.edges = legit_edges

    def output(self) -> str:
        """Outputs a list of edges to be inserted into blockMeshDict"""
        s = "edges\n(\n"

        for edge in self.edges:
            if edge.kind == "line":
                continue

            if edge.kind == "project":
                point_list =  f"({edge.points})"
            elif edge.kind == "arc":
                point_list = constants.vector_format(edge.points)
            else:
                point_list = "(" + " ".join([constants.vector_format(p) for p in edge.points]) + ")"


            s += f"\t{edge.kind} {edge.vertex_1.mesh_index} {edge.vertex_2.mesh_index} {point_list}\n"

        s += ");\n\n"

        return s

    def __getitem__(self, index):
        return self.edges[index]

    def __len__(self):
        return len(self.edges)