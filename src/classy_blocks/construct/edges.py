import abc

from typing import List, Optional, Union, Callable

import numpy as np

from classy_blocks.types import VectorType, PointType, PointListType, EdgeKindType
from classy_blocks.base.transformable import TransformableBase
from classy_blocks.util import functions as f
from classy_blocks.util import constants

class EdgeData(TransformableBase, abc.ABC):
    """Common operations on classes for edge creation"""
    kind:EdgeKindType # Edge type, the string that follows vertices in blockMeshDict.edges

    def transform(self, function: Callable) -> 'EdgeData':
        """An arbitrary transform of this edge by a specified function"""
        return self

    def translate(self, displacement: VectorType) -> 'EdgeData':
        """Move all points in the edge (but not start and end)
        by a displacement vector."""
        displacement = np.asarray(displacement, dtype=constants.DTYPE)

        return self.transform(lambda p: p + displacement)

    def rotate(self, angle: float, axis: VectorType, origin: Optional[PointType] = None) -> 'EdgeData':
        """Rotates all points in this edge (except start and end Vertex) around an
        arbitrary axis and origin (be careful with projected edges, geometry isn't rotated!)"""
        return self.transform(lambda p: f.rotate(p, axis, angle, origin))

    def scale(self, ratio: float, origin: Optional[PointType] = None) -> 'EdgeData':
        """Scales the edge points around given origin"""
        return self.transform(lambda p: f.scale(p, ratio, origin))

class Line(EdgeData):
    """A 'line' edge is created by default and needs no extra parameters"""
    kind = 'line'


class Arc(EdgeData):
    """Parameters for an arc edge: classic OpenFOAM circular arc
    definition with a single point lying anywhere on the arc"""
    kind = 'arc'

    def __init__(self, arc_point:PointType):
        self.point = np.asarray(arc_point, dtype=constants.DTYPE)

    def transform(self, function: Callable) -> 'Arc':
        self.point = function(self.point)
        return self

class Origin(EdgeData):
    """Parameters for an arc edge, alternative ESI-CFD version;
    defined with an origin point and optional flatness (default 1)

    https://www.openfoam.com/news/main-news/openfoam-v20-12/pre-processing#x3-22000
    https://develop.openfoam.com/Development/openfoam/-/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.H

    All arc variants are supported by classy_blocks;
    however, only the first (classic) one will be written to blockMeshDict for compatibility.
    If an edge was specified by 'angle' or 'origin', the definition will be output as a comment
    next to that edge definition."""
    kind = 'origin'

    def __init__(self, origin:PointType, flatness:float=1):
        self.origin = np.asarray(origin, dtype=constants.DTYPE)
        self.flatness = flatness

    def transform(self, function: Callable) -> 'Origin':
        self.origin = function(self.origin)
        return self

class Angle(EdgeData):
    """Parameters for an arc edge, alternative definition
    by Foundation (.org); defined with sector angle and axis
    
    https://github.com/OpenFOAM/OpenFOAM-10/commit/73d253c34b3e184802efb316f996f244cc795ec6

    All arc variants are supported by classy_blocks;
    however, only the first (classic) one will be written to blockMeshDict for compatibility.
    If an edge was specified by 'angle' or 'origin', the definition will be output as a comment
    next to that edge definition."""
    kind = 'angle'

    def __init__(self, angle:float, axis:VectorType):
        self.angle = angle
        self.axis = f.unit_vector(axis)

    def transform(self, function: Callable) -> 'Angle':
        self.axis = function(self.axis)
        self.axis = f.unit_vector(self.axis)
        return self

class Spline(EdgeData):
    """Parameters for a spline edge"""
    kind = 'spline'

    def __init__(self, points:PointListType):
        self.points = np.asarray(points, dtype=constants.DTYPE)

    def transform(self, function: Callable) -> 'Spline':
        self.points = np.asarray([function(p) for p in self.points])
        return self

class PolyLine(Spline):
    """Parameters for a polyLine edge"""
    kind = 'polyLine'

class Project(EdgeData):
    """Parameters for a 'project' edge"""
    kind = 'project'

    def __init__(self, geometry:Union[str, List[str]]):
        if isinstance(geometry, list):
            self.geometry = geometry
        else:
            self.geometry = [geometry]
