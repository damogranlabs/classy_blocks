from typing import Type

from classy_blocks.base.exceptions import EdgeNotFoundError
from classy_blocks.items.vertex import Vertex
from classy_blocks.items.edges.edge import Edge

# TODO: make this automatic
from classy_blocks.items.edges.line import LineEdge
from classy_blocks.items.edges.arcs.arc import ArcEdge
from classy_blocks.items.edges.arcs.origin import OriginEdge
from classy_blocks.items.edges.arcs.angle import AngleEdge
from classy_blocks.items.edges.spline import SplineEdge, PolyLineEdge
from classy_blocks.items.edges.project import ProjectEdge

class EdgeFactory:
    """Generates edges as requested by the user or returns existing ones
    if they are defined already"""
    def __init__(self):
        self.kinds = {}
        self.registry = []

    def find(self, vertex_1:Vertex, vertex_2:Vertex) -> Edge:
        """Returns an existing edge between given vertices
        or raises an EdgeNotFoundError otherwise"""
        indexes = {vertex_1.index, vertex_2.index}

        for edge in self.registry:
            if indexes == {edge.vertex_1.index, edge.vertex_2.index}:
                return edge
        
        raise EdgeNotFoundError(f"Edge between vertices not found: {indexes}")

    def register_kind(self, creator: Type[Edge]) -> None:
        """Introduces a new edge kind to this factory"""
        self.kinds[creator.kind] = creator

    def create(self, *args, duplicate=False):
        """Creates an EdgeOps of the desired kind and returns it"""
        # all definitions begin with
        # vertex_1: Vertex
        # vertex_2: Vertex

        vertex_1 = args[0]
        vertex_2 = args[1]

        def create_new() -> Edge:
            arg = list(args)
            kind = arg.pop(2)
            edge_class = self.kinds[kind]
            edge = edge_class(*arg)

            if kind != 'line':
                # do not include line edges;
                # they're here only for convenience (.length() and whatnot)
                self.registry.append(edge)
        
            return edge
            
        if duplicate:
            return create_new()

        try:
            return self.find(vertex_1, vertex_2)
        except EdgeNotFoundError:
            return create_new()

factory = EdgeFactory()
factory.register_kind(LineEdge)
factory.register_kind(ArcEdge)
factory.register_kind(OriginEdge)
factory.register_kind(AngleEdge)
factory.register_kind(SplineEdge)
factory.register_kind(PolyLineEdge)
factory.register_kind(ProjectEdge)
