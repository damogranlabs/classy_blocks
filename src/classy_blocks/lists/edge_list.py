from typing import List, Tuple, Literal, Optional, TypeAlias

from classy_blocks.base.exceptions import EdgeNotFoundError
from classy_blocks.construct.edges import EdgeData, Project, EdgeInfo
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.operations.projections import ProjectedEdgeData
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

        raise EdgeNotFoundError(f"Edge not found: {str(vertex_1)}, {str(vertex_2)}")

    def add(self, vertex_1:Vertex, vertex_2:Vertex, data:EdgeData) -> Edge:
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

    def add_from_operation(self, vertices:List[Vertex], operation:Operation) -> List[Edge]:
        """Queries the operation for edge data and creates edge objects from it"""
        edges = self.add_projected(vertices, operation.projections.edges)

        edges = \
            self.add_from_direction(vertices, operation.bottom_face.edges, 'bottom') + \
            self.add_from_direction(vertices, operation.top_face.edges, 'top') + \
            self.add_from_direction(vertices, operation.side_edges, 'side')

        return edges

    def add_from_direction(self,
            vertices:List[Vertex], # 8 vertices of the block
            data:List[Optional[EdgeData]], # a list of edges from Face/Operation
            direction:Literal['bottom', 'top', 'side'] # use indexes for top/bottom face, or sides
            ) -> List[EdgeInfo]:
        """Collect edges from faces and between them;
        check for duplicates (same vertex pairs) and validity
        (no lines or straight-line arcs); return created edges so that
        (other) blocks get to know them"""

        edges:List[EdgeInfo] = []

        for corner_1, edge_data in enumerate(data):
            if edge_data is None:
                continue

            corner_1, corner_2 = self.get_corners(corner_1, direction)

            vertex_1 = vertices[corner_1]
            vertex_2 = vertices[corner_2]

            edge = self.add(vertex_1, vertex_2, edge_data)
            edges.append((corner_1, corner_2, edge))

        return edges

    def add_projected(self, vertices:List[Vertex], data:List[ProjectedEdgeData]) -> List[EdgeInfo]:
        """Collect projected edge data from operation"""
        # TODO: test
        edges = []

        for edge_data in data:
            vertex_1 = vertices[edge_data.corner_1]
            vertex_2 = vertices[edge_data.corner_2]

            try:
                # remove existing edges between those vertices
                self.edges.remove(self.find(vertex_1, vertex_2))
            except EdgeNotFoundError:
                # no such edge yet, ok
                pass

            edge = factory.create(vertex_1, vertex_2, Project(edge_data.geometry))
            self.edges.append(edge)
            edges.append(EdgeInfo(edge_data.corner_1, edge_data.corner_2, edge))

        return edges

    @property
    def description(self) -> str:
        """Outputs a list of edges to be inserted into blockMeshDict"""
        out = "edges\n(\n"

        for edge in self.edges:
            out += f"\t{edge.description}\n"

        out += ");\n\n"

        return out

    @staticmethod
    def get_corners(corner_1, direction:Literal['bottom', 'top', 'side']) -> Tuple[int, int]:
        """returns indexes of the first and second corner
        with respect to given direction:
        bottom: 0-1, 1-2, 2-3, 3-0
        top: 4-5, 5-6, 6-7, 7-4
        side: 0-4, 1-5, 2-6, 3-7"""
        # match direction: # once in the future
        #     case 'bottom':
        #         corner_2 = (corner_1+1)%4
        #     case 'top':
        #         corner_1 = 4 + corner_1
        #         corner_2 = 4 + (corner_1+1)%4
        #     case 'side':
        #         corner_2 = 4 + corner_1
        #     case other:
        #         raise ValueError(f"No defined direction: {direction}")

        if direction ==  'bottom':
            return corner_1, (corner_1+1)%4

        if direction == 'top':
            return corner_1 + 4, (corner_1+1)%4 + 4

        if direction == 'side':
            return corner_1, corner_1 + 4

        raise ValueError(f"No defined direction: {direction}")
