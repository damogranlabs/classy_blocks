from typing import List, Callable

import numpy as np
import copy

from util.methematics import functions as g
from util import constants, tools

from classes.primitives import Vertex, Edge
from classes.block import Block

def transform_points(points, function):
    return [function(p) for p in points]

def transform_edges(edges, function):
    # edges: some are 
    if edges is not None:
        new_edges = [None]*4
        for i, edge_points in enumerate(edges):
            edge_type, edge_points = Edge.get_type(edge_points)

            if edge_type == 'spline':
                new_edges[i] = [function(e) for e in edge_points]
            elif edge_type == 'arc':
                new_edges[i] = function(edge_points)

        return new_edges
    
    return None

class Face:
    def __init__(self, points:List[List[float]], edges:List[Edge]=None, check_coplanar:bool=True):
        """ a Face is a collection of 4 points and optionally 4 edge points; """
        if len(points) != 4:
            raise Exception("Provide exactly 4 points")

        self.points = np.array(points)

        if edges is not None:
            if len(edges) != 4:
                raise Exception("Exactly four edges must be supplied - use None for straight lines")

        self.edges = edges

        if check_coplanar:
            x = self.points
            if abs(np.dot((x[1] - x[0]), np.cross(x[3] - x[0], x[2] - x[0]))) > constants.tol:
                raise Exception("Points are not coplanar!")
        
            # TODO: coplanar edges?
    
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

    def transform(self, function):
        # TODO: typing
        """ copis this object and transforms all block-defining points
        with given function. used for translation, rotating, etc. """
        t = copy.copy(self)
        t.points = transform_points(self.points, function)
        t.edges = transform_edges(self.edges, function)

        return t

    def translate(self, vector):
        # TODO: typing 
        """ move points by 'vector' """
        vector = np.array(vector)
        return self.transform(lambda point: point + vector)

    def rotate(self, axis, angle, origin=[0, 0, 0]):
        # TODO: typing
        """ copies the object and rotates all block-points by 'angle'
        around 'axis' going through 'origin' """
        axis = np.array(axis)
        origin = np.array(origin)

        r = lambda point: g.arbitrary_rotation(point, axis, angle, origin)

        return self.transform(r)


class Operation():
    def __init__(self, bottom_face:Face, top_face:Face, side_edges:List[Edge]=None):
        self.bottom_face = bottom_face
        self.top_face = top_face

        # edges between vertices on same face
        self.edges = self.bottom_face.get_edges() + self.top_face.get_edges(top_face=True)

        # edges between same vertices between faces
        self.side_edges = side_edges

        if self.side_edges is not None:
            if len(self.side_edges) != 4:
                raise Exception("Provide 4 edges for sides; use None for straight lines")

            for i in range(4):
                e = self.side_edges[i]

                if e is not None:
                    self.edges.append(Edge(i, i+4, e))

        # create a block and edges
        self.block = Block.create_from_points(
            np.concatenate((bottom_face.points, top_face.points)),
            self.edges
        )

    def set_cell_count(self, axis, count):
        """ Directly set number of cells for given axis:
         0: along first edge of a face
         1: along second edge of a face
         2: between faces / along operation path
        """
        self.block.n_cells[axis] = int(count)

    def count_to_size(self, axis, cell_size):
        """ Calculate cell count to meet cell_size.
        Axes:
         0: along first edge of a face
         1: along second edge of a face
         2: between faces / along operation path
        """
        return self.block.count_to_size(axis, cell_size)

    def grade_to_size(self, axis, size):
        """ Sets block grading so that the final cell size is as required;
        to set that cell size on opposite (beginning) side, use negative size.

        Axes:
         0: along first edge of a face
         1: along second edge of a face
         2: between faces / along operation path """
        self.block.grade_to_size(axis, abs(size), size<0)

    def set_patch(self, sides, patch_name):
        # TODO: typing
        """ bottom: bottom face
        top: top face

        front: along first edge of a face
        back: opposite front

        right: along second edge of a face
        left: opposite right """
        self.block.set_patch(sides, patch_name)

    def translate(self, vector):
        # TODO: test
        """ returns a translated copy of this Operation """
        vector = np.array(vector)
        
        bottom_face = self.bottom_face.translate(vector)
        top_face = self.top_face.translate(vector)

        side_edges = transform_edges(self.side_edges, lambda v: v + vector)

        return Operation(bottom_face, top_face, side_edges)
                
    def rotate(self, axis, angle, origin=[0, 0, 0]):
        # TODO: typing
        # TODO: test
        axis = np.array(axis)
        origin = np.array(origin)

        bottom_face = self.bottom_face.rotate(axis, angle, origin)
        top_face = self.top_face.rotate(axis, angle, origin)

        side_edges = transform_edges(
            self.side_edges,
            lambda point: g.arbitrary_rotation(point, axis, angle, origin)
        )

        return Operation(bottom_face, top_face, side_edges)
