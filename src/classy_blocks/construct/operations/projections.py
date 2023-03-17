"""Auxiliary dataclasses for storing operation's projected  entities"""
import dataclasses

from typing import List, Dict

from classy_blocks.types import ProjectToType, OrientType

@dataclasses.dataclass
class ProjectedEdgeData:
    """Data for a projected edge"""
    corner_1:int
    corner_2:int
    geometry:ProjectToType

@dataclasses.dataclass
class ProjectedVertexData:
    """Data for a projected Vertex"""
    corner:int
    geometry:ProjectToType

@dataclasses.dataclass
class ProjectedEntities:
    """A database of operation's projected entities"""
    sides:Dict[OrientType, str] = dataclasses.field(default_factory=dict)
    edges:List[ProjectedEdgeData] = dataclasses.field(default_factory=list)
    vertices:List[ProjectedVertexData] = dataclasses.field(default_factory=list)

    def add_side(self, side:OrientType, geometry:str) -> None:
        """Add a projected side to database"""
        self.sides[side] = geometry

    def add_edge(self, corner_1:int, corner_2:int, geometry:ProjectToType) -> None:
        """Add a projected edge to database"""
        self.edges.append(ProjectedEdgeData(corner_1, corner_2, geometry))

    def add_vertex(self, corner:int, geometry:ProjectToType) -> None:
        """Add a projected vertex to database"""
        self.vertices.append(ProjectedVertexData(corner, geometry))
