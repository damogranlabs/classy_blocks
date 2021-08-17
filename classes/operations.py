from typing import List
import copy

import numpy as np

from ..util import functions as f
from ..util import constants

from .block import Block
from .primitives import Edge

def transform_points(points, function):
    return [function(p) for p in points]

def transform_edges(edges, function):
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
    def __init__(self, points:List[List[float]], edges:List[Edge]=None, check_coplanar:bool=False):
        """ a Face is a collection of 4 points and optionally 4 edge points; """
        if len(points) != 4:
            raise Exception("Provide exactly 4 points")

        self.points = np.array(points)

        if edges is not None:
            if len(edges) != 4:
                raise Exception("Exactly four Edges must be supplied - use None for straight lines")

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

    def set_patch(self, sides, patch_name:str):
        """ bottom: bottom face
        top: top face

        front: along first edge of a face
        back: opposite front

        right: along second edge of a face
        left: opposite right """
        self.block.set_patch(sides, patch_name)

    def translate(self, vector:List):
        """ returns a translated copy of this Operation """
        vector = np.array(vector)
        
        bottom_face = self.bottom_face.translate(vector)
        top_face = self.top_face.translate(vector)

        side_edges = transform_edges(self.side_edges, lambda v: v + vector)

        return Operation(bottom_face, top_face, side_edges)
                
    def rotate(self, axis:List, angle:float, origin:List=[0, 0, 0]):
        axis = np.array(axis)
        origin = np.array(origin)

        bottom_face = self.bottom_face.rotate(axis, angle, origin)
        top_face = self.top_face.rotate(axis, angle, origin)

        side_edges = transform_edges(
            self.side_edges,
            lambda point: f.arbitrary_rotation(point, axis, angle, origin)
        )

        return Operation(bottom_face, top_face, side_edges)

    def set_cell_zone(self, cell_zone):
        self.block.cell_zone = cell_zone


class Loft(Operation):
    """ since any possible block shape can be created with Loft operation,
    Loft is the most low-level of all operations. Anything included in Loft
    must also be included in Operation. """
    pass

class Extrude(Loft):
    """ Takes a Face and extrudes it in given extrude_direction """
    def __init__(self, base:Face, extrude_vector:list):
        self.base = base
        self.extrude_vector = extrude_vector

        top_face = base.translate(self.extrude_vector)

        super().__init__(base, top_face)

class Revolve(Loft):
    def __init__(self, base:Face, angle:list, axis:list, origin:list):
        """ Takes a Face and revolves it by angle around axis;
        axis can be translated so that it goes through desired origin.
        
        Angle is given in radians, 
        revolve is in positive sense (counter-clockwise) """
        self.base = base
        self.angle = angle
        self.axis = axis
        self.origin = origin

        bottom_face = base
        top_face = base.rotate(axis, angle, origin)

        # there are 4 side edges: rotate each vertex of bottom_face
        # by angle/2
        side_edges = [
            f.arbitrary_rotation(p, self.axis, self.angle/2, self.origin)
            for p in self.base.points
        ]

        super().__init__(bottom_face, top_face, side_edges)

class Wedge(Revolve):
    def __init__(self, face:Face, angle=f.deg2rad(2)):
        """ Revolves 'face' around x-axis symetrically by +/- angle/2.
        By default, the angle is 2 degrees.

        Used for creating wedge-type geometries for axisymmetric cases.
        Automatically creates wedge patches* (you still
        need to include them in changeDictionaryDict - type: wedge).

        * - default naming of block sides is not very intuitive
        for wedge geometry so additional methods are available for wedges:
            set_outer_patch,
            set_inner_patch,
            set_left_patch,
            set_right_patch,
        other two patches are wedge_left and wedge_right. Sides are named
        according to this sketch:

                          outer
            _________________________________
            |                               |
            | left                    right |
            |_______________________________|
                          inner
        __  _____  __  _____  __  _____  __  __ axis of symmetry (x) """
        
        # default axis
        axis = [1, 0, 0]
        # default origin
        origin = [0, 0, 0]

        # first, rotate this face forward, then use init this as Revolve
        # and rotate the same face
        base = face.rotate(axis, -angle/2, origin)

        super().__init__(base, angle, axis, origin)

        # assign 'wedge_left' and 'wedge_right' patches
        super().set_patch('top', 'wedge_front')
        super().set_patch('bottom', 'wedge_back')

        # there's also only 1 cell in z-direction
        self.set_cell_count(2, 1)

    def set_patch(self):
        raise NotImplementedError("Use set_[outer|inner|left|right]_patch methods for wedges")

    def set_outer_patch(self, patch_name):
        super().set_patch('back', patch_name)
    
    def set_inner_patch(self, patch_name):
        super().set_patch('front', patch_name)

    def set_left_patch(self, patch_name):
        super().set_patch('left', patch_name)

    def set_right_patch(self, patch_name):
        super().set_patch('right', patch_name)
    
