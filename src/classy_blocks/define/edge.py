import warnings

from typing import List, Callable, Union, Tuple, Optional
from classy_blocks.types import PointType, VectorType, EdgeKindType, EdgePointsType

import numpy as np

from classy_blocks.define.vertex import Vertex

from classy_blocks.util import functions as f
from classy_blocks.util import constants

def arc_mid(axis:VectorType, center:PointType,
    radius:float, edge_point_0:PointType, edge_point_1:PointType) -> PointType:
    """Returns the midpoint of the specified arc in 3D space"""
    # Kudos to this shrewd solution
    # https://math.stackexchange.com/questions/3717427
    axis = np.asarray(axis)
    edge_point_0 = np.asarray(edge_point_0)
    edge_point_1 = np.asarray(edge_point_1)

    sec = edge_point_1 - edge_point_0
    sec_ort = np.cross(sec, axis)

    return center + f.unit_vector(sec_ort) * radius

def arc_from_theta(edge_point_0:PointType, edge_point_1:PointType, angle:float, axis:VectorType) -> PointType:
    """Calculates a point on the arc edge from given sector angle and an
    axis of the arc. An interface to the Foundation's
    arc <vertex-0> <vertex-1> <angle> (centerpoint) alternative edge specification:
    https://github.com/OpenFOAM/OpenFOAM-dev/commit/73d253c34b3e184802efb316f996f244cc795ec6"""
    # Meticulously transcribed from
    # https://github.com/OpenFOAM/OpenFOAM-dev/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.C

    assert 0 < angle < 360, f"Angle {angle} should be between 0 and 2*pi"

    axis = np.asarray(axis)
    edge_point_0 = np.asarray(edge_point_0)
    edge_point_1 = np.asarray(edge_point_1)

    dp = edge_point_1 - edge_point_0

    pM = (edge_point_0 + edge_point_1)/2
    rM = f.unit_vector(np.cross(dp, axis))

    l = np.dot(dp, axis)

    chord = dp - l*axis
    magChord = f.norm(chord)

    center = pM - l*axis/2 - rM*magChord/2/np.tan(angle/2)
    radius = f.norm(edge_point_0 - center)
    
    return arc_mid(axis, center, radius, edge_point_0, edge_point_1)

def arc_from_origin(edge_point_0:PointType, edge_point_1:PointType, center:PointType,
    adjust_center:bool=True, r_multiplier:float=1.0):
    """Calculates a point on the arc edge from given endpoints and arc origin.
    An interface to ESI-CFD's alternative arc edge specification:
    https://www.openfoam.com/news/main-news/openfoam-v20-12/pre-processing#pre-processing-blockmesh"""
    # meticulously transcribed from 
    # https://develop.openfoam.com/Development/openfoam/-/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.C
    
    # Position vectors from centre
    p1 = edge_point_0
    p3 = edge_point_1

    r1 = p1 - center
    r3 = p3 - center

    mag1 = f.norm(r1)
    mag3 = f.norm(r3)

    chord = p3 - p1

    axis = np.cross(r1, r3)

    # The average radius
    radius = 0.5*(mag1 + mag3)

    # The included angle (not needed)
    # angle = np.arccos(np.dot(r1, r3)/(mag1*mag3))

    needs_adjust = False

    if adjust_center:
        needs_adjust = abs(mag1 - mag3) > constants.tol

        if r_multiplier != 1:
            # The min radius is constrained by the chord,
            # otherwise bad things will happen.
            needs_adjust = True
            radius = radius * r_multiplier
            radius = max(radius, (1.001*0.5*f.norm(chord)))

    if needs_adjust:
        # The centre is not equidistant to p1 and p3.
        # Use the chord and the arcAxis to determine the vector to
        # the midpoint of the chord and adjust the centre along this
        # line.
        new_center = (0.5 * (p3 + p1)) + \
            (radius**2 - 0.25 * f.norm(chord)**2)**0.5 * \
            f.unit_vector(np.cross(axis, chord)) # mid-chord -> centre

        warnings.warn("Adjusting center of edge between" +
            f" {str(edge_point_0)} and {str(edge_point_1)}")

        return arc_from_origin(p1, p3, new_center, False)
    
    # done, return the calculated point
    return arc_mid(axis, center, radius, edge_point_0, edge_point_1)


class Edge:
    """An Edge, defined by two vertices and points in between;
    a single point edge is treated as 'arc', more points are
    treated as 'spline'.

    passed indexes refer to position in Block.edges[] list; Mesh.write()
    will assign actual Vertex objects."""
    def __init__(self, block_index_1:int, block_index_2:int,
            points: EdgePointsType, kind:Optional[EdgeKindType]=None):
        # TODO
        # indexes can only be 1 vertex apart (edges within top/bottom face)
        # or 4 vertices apart (edges between the same corner in top and bottom face)
        
        # indexes in block.edges[] list
        self.block_index_1 = block_index_1
        self.block_index_2 = block_index_2

        # these will refer to actual Vertex objects after Mesh.write()
        self.vertex_1 = None
        self.vertex_2 = None

        assert points is not None
        self.points = np.asarray(points)

        # if not specified, determine the kind of this edge
        if kind is not None:
            self.kind = kind
        else:
            if isinstance(points, str):
                self.kind = 'project'
            else:
                # TODO: check for proper array dimensions

                # arc edges are 1d-arrays (single point)
                # others are 2d (lists of points)
                shape = np.shape(self.points)
                if len(shape) == 1:
                    self.kind = 'arc'
                else:
                    # the default for a list of points
                    self.kind = 'spline'

    @property
    def is_valid(self):
        # wedge geometries produce coincident
        # edges and vertices; drop those
        if f.norm(self.vertex_1.point - self.vertex_2.point) < constants.tol:
            return False

        # 'all' spline and projected edges are 'valid'
        if self.kind in ("line", "spline", "project"):
            return True

        # if case vertex1, vertex2 and point in between
        # are collinear, blockMesh will find an arc with
        # infinite radius and crash.
        # so, check for collinearity; if the three points
        # are actually collinear, this edge is redundant and can be
        # silently dropped
        OA = self.vertex_1.point
        OB = self.vertex_2.point
        OC = self.points

        # if point C is on the same line as A and B:
        # OC = OA + k*(OB-OA)
        AB = OB - OA
        AC = OC - OA

        k = f.norm(AC) / f.norm(AB)
        d = f.norm((OA + AC) - (OA + k * AB))

        return d > constants.tol

    def get_length(self) -> float:
        """Returns an approximate length of this Edge"""
        if self.kind in ("line", "project"):
            return f.norm(self.vertex_1.point - self.vertex_2.point)

        if self.kind == "arc":
            return f.arc_length_3point(self.vertex_1.point, self.points, self.vertex_2.point)

        def curve_length(points):
            l = 0

            for i in range(len(points) - 1):
                l += f.norm(points[i + 1] - points[i])

            return l

        # other edges are a bunch of points
        edge_points = np.concatenate(([self.vertex_1.point], self.points, [self.vertex_2.point]), axis=0)
        return curve_length(edge_points)


    def translate(self, displacement:VectorType):
        """Move all points in the edge (but not start and end) 
        by a displacement vector."""
        displacement = np.asarray(displacement)
        
        if self.kind == 'arc':
            new_points = self.points + displacement
        elif self.kind != 'project':
            new_points = [e + displacement for e in self.points]
        # project and line edges don't require any modifications
        
        return self.__class__(self.block_index_1, self.block_index_2, new_points, self.kind)

    def rotate(self, angle:List[float], axis:List[float], origin:List[float]=None):
        """ Rotates all points in this edge (except start and end Vertex) around an
        arbitrary axis and origin (be careful with projected edges, geometry isn't rotated!)"""
        if origin is None:
            origin = [0, 0, 0]

        # TODO: include/exclude projected edges?
        # TODO: optimize
        if self.kind == 'project':
            points = self.points
        elif self.kind == 'arc':
            points = f.arbitrary_rotation(self.points, axis, angle, origin)
        else:
            points = [f.arbitrary_rotation(p, axis, angle, origin) for p in self.points]
        
        return self.__class__(self.block_index_1, self.block_index_2, points, kind=self.kind)

    def scale(self, ratio:float, origin):
        """Scales the edge points around given origin"""
        if self.kind == 'project':
            points = self.points
        elif self.kind == 'arc':
            points = Vertex.scale_point(self.points, ratio, origin)
        else:
            points = [Vertex.scale_point(p, ratio, origin) for p in self.points]
        
        return self.__class__(self.block_index_1, self.block_index_2, points, kind=self.kind)