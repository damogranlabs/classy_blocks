from typing import Type

from classy_blocks.items.edges.edge import Edge

# TODO: make this automatic
from classy_blocks.items.edges.line import LineEdge
from classy_blocks.items.edges.arcs.arc import ArcEdge
from classy_blocks.items.edges.arcs.origin import OriginEdge
from classy_blocks.items.edges.arcs.angle import AngleEdge
from classy_blocks.items.edges.spline import SplineEdge, PolyLineEdge
from classy_blocks.items.edges.project import ProjectEdge

class EdgeFactory:
    """Kind of a pattern"""

    def __init__(self):
        self.kinds = {}

    def register_kind(self, creator: Type[Edge]) -> None:
        """Introduces a new edge kind to this factory"""
        self.kinds[creator.kind] = creator

    def create(self, *args):
        """Creates an EdgeOps of the desired kind and returns it"""
        # all definitions begin with
        # vertex_1: Vertex
        # vertex_2: Vertex
        args = list(args)
        kind = self.kinds[args.pop(2)]
        return kind(*args)

factory = EdgeFactory()
factory.register_kind(LineEdge)
factory.register_kind(ArcEdge)
factory.register_kind(OriginEdge)
factory.register_kind(AngleEdge)
factory.register_kind(SplineEdge)
factory.register_kind(PolyLineEdge)
factory.register_kind(ProjectEdge)
