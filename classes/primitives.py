# -*- coding: utf-8 -*-
import numpy as np

from ..util import functions as f
from ..util import constants

# see README for terminology, terminolology, lol
class Vertex():
    """ point with an index that's used in block and face definition
    and can output in OpenFOAM format """
    def __init__(self, point):
        self.point = np.array(point)
        self.mesh_index = None # will be changed in Mesh.prepare_data()
        
    def rotate(self, angle, axis=[1, 0, 0], origin=[0, 0, 0]):
        """ returns a new, rotated Vertex """
        point = f.arbitrary_rotation(self.point, axis, angle, origin)
        return Vertex(point)

    def __repr__(self):
        s = constants.vector_format(self.point)
        
        if self.mesh_index is not None:
            s += " // {}".format(self.mesh_index)
        
        return s

    def rotate(self, angle, axis=[1, 0, 0], origin=[0, 0, 0]):
        """ returns a new, rotated Vertex """
        point = f.arbitrary_rotation(self.point, axis, angle, origin)
        return Vertex(point)

class Edge():
    def __init__(self, index_1, index_2, points):
        """ an edge is defined by two vertices and points in between;
        a single point edge is treated as 'arc', more points are
        treated as 'spline'.

        passed indexes refer to position in Block.edges[] list; Mesh.prepare_data()
        will assign actual Vertex objects.
        """

        # indexes in block.edges[] list
        self.block_index_1 = index_1
        self.block_index_2 = index_2

        # these will refer to actual Vertex objects after Mesh.prepare_data()
        self.vertex_1 = None
        self.vertex_2 = None

        self.type, self.points = self.get_type(points)

    @staticmethod
    def get_type(points):
        """ returns edge type and a list of points:
        'None' for a straight line,
        'arc' for a circular arc,
        'spline' for a spline """

        if points is None:
            return None, None

        # if multiple points are given check that they are of correct length
        points = np.array(points)
        shape = np.shape(points)

        if len(shape) == 1:
            t = 'arc'
        else:
            assert len(shape) == 2
            for p in points:
                assert len(p) == 3
            
            t = 'spline'

        return t, points

    @property
    def point_list(self):
        if self.type == 'arc':
            return constants.vector_format(self.points)
        else:
            return "(" +  \
                   " ".join([constants.vector_format(p) for p in self.points]) + \
                   ")"
    
    @property
    def is_valid(self):
        # 'all' spline edges are 'valid'
        if self.type == 'spline':
            return True
        
        # wedge geometries produce coincident 
        # edges and vertices; drop those
        if f.norm(self.vertex_1.point - self.vertex_2.point) < constants.tol:
            return False

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

        k = f.norm(AC)/f.norm(AB)
        d = f.norm((OA+AC) - (OA + k*AB))

        return d > constants.tol

    def get_length(self):
        def curve_length(points):
            l = 0

            for i in range(len(points)-1):
                l += f.norm(points[i+1] - points[i])

            return l

        if self.type == 'arc':
            edge_points = np.array([
                self.vertex_1.point,
                self.points,
                self.vertex_2.point
            ])

            return curve_length(edge_points)
        elif self.type == 'spline':
            edge_points = np.concatenate((
                [self.vertex_1.point],
                self.points,
                [self.vertex_2.point]), axis=0)
            return curve_length(edge_points)
        else:
            raise AttributeError(f"Unknown edge type: {self.type}")

    def rotate(self, angle, axis=[1, 0, 0], origin=[0, 0, 0]):
        if self.type == 'arc':
            points = f.arbitrary_rotation(self.points, axis, angle, origin)
        else:
            points = [f.arbitrary_rotation(p, axis, angle, origin) for p in self.points]
        
        return Edge(self.block_index_1, self.block_index_2, points)
        
    def __repr__(self):
        return "{} {} {} {}".format(
            self.type,
            self.vertex_1.mesh_index,
            self.vertex_2.mesh_index,
            self.point_list
        )