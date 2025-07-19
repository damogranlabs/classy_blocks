from classy_blocks.base.exceptions import EdgeNotFoundError
from classy_blocks.construct.edges import EdgeData
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.edges.factory import factory
from classy_blocks.items.vertex import Vertex

EdgeLocationType = tuple[int, int]


def get_location(vertex_1, vertex_2):
    if vertex_1.index < vertex_2.index:
        return (vertex_1.index, vertex_2.index)

    return (vertex_2.index, vertex_1.index)


class EdgeList:
    """Handling of the 'edges' part of blockMeshDict"""

    def __init__(self) -> None:
        self.edges: dict[EdgeLocationType, Edge] = {}

    def find(self, vertex_1: Vertex, vertex_2: Vertex) -> Edge:
        location = get_location(vertex_1, vertex_2)

        if location in self.edges:
            return self.edges[location]

        raise EdgeNotFoundError

    def add(self, vertex_1: Vertex, vertex_2: Vertex, data: EdgeData) -> Edge:
        """Adds an edge between given vertices or returns an existing one"""
        location = get_location(vertex_1, vertex_2)

        # if this edge exists in the list, return it regardless of what's
        # specified in edge_data; redefinitions of the same edges are ignored
        if location in self.edges:
            return self.edges[location]

        edge = factory.create(vertex_1, vertex_2, data)

        if edge.is_valid:
            self.edges[location] = edge

        return edge

    def add_from_operation(self, vertices: list[Vertex], operation: Operation) -> list[tuple[int, int, Edge]]:
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
