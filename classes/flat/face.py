from typing import List
import copy

import numpy as np

from ..primitives import Edge, transform_edges, transform_points
from ...util import constants
from ...util import functions as f

class Face:
    def __init__(self, points:List[List[float]], edges:List[Edge]=None, check_coplanar:bool=False):
        """ a Face is a collection of 4 points and optionally 4 edge points; """
        if len(points) != 4:
            raise Exception("Provide exactly 4 points")

        self.points = np.asarray(points)

        if edges is not None:
            if len(edges) != 4:
                raise Exception("Exactly four Edges must be supplied - use None for straight lines")

        self.edges = edges

        if check_coplanar:
            x = self.points
            if abs(np.dot((x[1] - x[0]), np.cross(x[3] - x[0], x[2] - x[0]))) > constants.tol:
                raise Exception("Points are not coplanar!")
        
            # TODO: coplanar edges?
        
        # center and normal
        self.normal, self.center = self.get_normal(self.points)
        
    def get_edges(self, top_face:bool=False) -> List[Edge]:
        if not self.edges:
            return []

        r = [] # a list of Edge objects will be returned

        # correct vertex index so that it refers to block numbering
        # (bottom face vertices 0...3, top face 4...7)
        for i in range(4):
            if self.edges[i] is not None:
                # bottom face indexes: 0 1 2 3
                i_1 = i
                i_2 = (i+1)%4

                if top_face:
                    # top face indexes: 4 5 6 7
                    i_1 += 4
                    i_2 = (i_1+1)%4 + 4

                r.append(Edge(i_1, i_2, self.edges[i%4]))

        return r

    def _transform(self, function):
        """ copis this object and transforms all block-defining points
        with given function. used for translation, rotating, etc. """
        t = copy.copy(self)
        t.points = transform_points(self.points, function)
        t.edges = transform_edges(self.edges, function)

        return t

    def translate(self, vector:List):
        """ move points by 'vector' """
        vector = np.array(vector)
        return self._transform(lambda point: point + vector)

    def rotate(self, axis:List, angle:float, origin:List=[0, 0, 0]):
        """ copies the object and rotates all block-points by 'angle'
        around 'axis' going through 'origin' """
        axis = np.array(axis)
        origin = np.array(origin)

        r = lambda point: f.arbitrary_rotation(point, axis, angle, origin)

        return self._transform(r)

    def scale(self, ratio:float, origin:List=None):
        """ Scale with respect to given origin """
        if origin is None:
            origin = self.center

        r = lambda point: origin + (point - origin)*ratio

        return self._transform(r)

    def invert(self):
        """ reverses the order of points """
        self.points = np.flip(self.points, axis=0)

    @staticmethod
    def get_normal(points):
        # calculate face normal; OpenFOAM divides a quadrangle into 4 triangles,
        # each joining at face center; a normal is an average of normals of those
        # triangles
        # TODO: TEST
        points = np.asarray(points)
        center = np.average(points, axis=0)
        side_1 = points - center
        side_2 = np.roll(points, -1, axis=0) - center
        normals = np.cross(side_1, side_2)
        normal = np.average(normals, axis=0)

        return normal, center
    