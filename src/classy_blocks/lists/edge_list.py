from typing import List, Tuple

from classy_blocks.base.exceptions import EdgeNotFoundError
from classy_blocks.construct.edges import EdgeData
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.factory import factory
from classy_blocks.items.vertex import Vertex


class EdgeList:
    """Handling of the 'edges' part of blockMeshDict"""

    def __init__(self) -> None:
        self.edges: List[Edge] = []

    def find(self, vertex_1: Vertex, vertex_2: Vertex) -> Edge:
        """checks if an edge with the same pair of vertices
        exists in self.edges already"""
        for edge in self.edges:
            if {vertex_1.index, vertex_2.index} == {edge.vertex_1.index, edge.vertex_2.index}:
                return edge

        raise EdgeNotFoundError(f"Edge not found: {vertex_1}, {vertex_2}")

    def add(self, vertex_1: Vertex, vertex_2: Vertex, data: EdgeData) -> Edge:
        """Adds an edge between given vertices or returns an existing one"""
        try:
            # if this edge exists in the list, return it regardless of what's
            # specified in edge_data; redefinitions of the same edges are ignored
            edge = self.find(vertex_1, vertex_2)
        except EdgeNotFoundError:
            edge = factory.create(vertex_1, vertex_2, data)

            if edge.is_valid:
                self.edges.append(edge)

        return edge

    def add_from_operation(self, vertices: List[Vertex], operation: Operation) -> List[Tuple[int, int, Edge]]:
        """Queries the operation for edge data and creates edge objects from it"""
        data_frame = operation.edges
        edges = []

        for data in data_frame.get_all_beams():
            corner_1 = data[0]
            corner_2 = data[1]

            vertex_1 = vertices[corner_1]
            vertex_2 = vertices[corner_2]

            edge = self.add(vertex_1, vertex_2, data[2])
            edges.append((corner_1, corner_2, edge))

        return edges

    def clear(self) -> None:
        """Empties all lists"""
        self.edges.clear()

    @property
    def description(self) -> str:
        """Outputs a list of edges to be inserted into blockMeshDict"""
        out = "edges\n(\n"

        for edge in self.edges:
            out += edge.description + "\n"

        out += ");\n\n"

        return out
