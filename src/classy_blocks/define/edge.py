from typing import List, Callable, Union, Tuple, Optional
from classy_blocks import types

import numpy as np

from classy_blocks.define.vertex import Vertex

from classy_blocks.util import functions as f
from classy_blocks.util import constants


class WrongEdgeTypeException(Exception):
    """Raised when using an unsupported edge type"""

    def __init__(self, edge_type, *args, **kwargs):
        raise Exception(f"Wrong edge type: {edge_type}", *args, **kwargs)

def transform_edges(edges: List["Edge"], function: Callable):
    """Same as transform_points but deals with multiple points in an edge, if applicable"""
    if edges is not None:
        new_edges = [None] * 4
        for i, edge in enumerate(edges):
            # spline edge is defined with multiple points,
            # a single point defines an arc;
            # 'project' and 'line' don't require any
            if edge.kind == "arc":
                new_edges[i] = function(edge.points)
            elif edge.kind != "project":
                new_edges[i] = [function(e) for e in edge.points]

        return new_edges

    return None


class Edge:
    """An Edge, defined by two vertices and points in between;
    a single point edge is treated as 'arc', more points are
    treated as 'spline'.

    passed indexes refer to position in Block.edges[] list; Mesh.write()
    will assign actual Vertex objects."""
    def __init__(self, block_index_1:int, block_index_2:int,
            points: types.EdgePointsType, kind:Optional[types.EdgeKindType]=None):
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


    def translate(self, displacement:types.VectorType):
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