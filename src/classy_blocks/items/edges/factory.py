from typing import Type

from classy_blocks.construct.edges import EdgeData
from classy_blocks.items.edges.arcs.angle import AngleEdge
from classy_blocks.items.edges.arcs.arc import ArcEdge
from classy_blocks.items.edges.arcs.origin import OriginEdge
from classy_blocks.items.edges.curve import OnCurveEdge, PolyLineEdge, SplineEdge
from classy_blocks.items.edges.edge import Edge

# FIXME: make this automatic
from classy_blocks.items.edges.line import LineEdge
from classy_blocks.items.edges.project import ProjectEdge
from classy_blocks.types import EdgeKindType


class EdgeFactory:
    """Generates edges as requested by the user or returns existing ones
    if they are defined already"""

    def __init__(self):
        self.kinds = {}

    def register_kind(self, kind: EdgeKindType, creator: Type[Edge]) -> None:
        """Introduces a new edge kind to this factory"""
        self.kinds[kind] = creator

    def create(self, vertex_1, vertex_2, data: EdgeData) -> Edge:
        """Creates an Edge* of the desired kind and returns it"""
        edge_class = self.kinds[data.kind]
        return edge_class(vertex_1, vertex_2, data)


factory = EdgeFactory()
factory.register_kind("line", LineEdge)
factory.register_kind("arc", ArcEdge)
factory.register_kind("origin", OriginEdge)
factory.register_kind("angle", AngleEdge)
factory.register_kind("spline", SplineEdge)
factory.register_kind("curve", OnCurveEdge)
factory.register_kind("polyLine", PolyLineEdge)
factory.register_kind("project", ProjectEdge)
