from typing import List

import numpy as np

from util import geometry as g
from util import constants, tools

from operations.base import Face, Operation

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
        axis can be translated so that it goes through origin.
        
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
            g.arbitrary_rotation(p, self.axis, self.angle/2, self.origin)
            for p in self.base.points
        ]

        super().__init__(bottom_face, top_face, side_edges)

class Wedge(Revolve):
    def __init__(self, face:Face, angle=g.deg2rad(5)):
        """ Revolves 'face' around x-axis symetrically by +/- angle/2.
        By default, angle is 2 degrees.

        Used for creating wedge-type geometries for axisymmetric cases.
        Automatically creates wedge patches* (you still
        need to include them in changeDictionaryDict - type: wedge).

        * - default naming of block sides is not very intuitive
        for wedge geometry so additional methods are available for wedges:
            set_outer_patch,
            set_core_patch,
            set_left_patch,
            set_right_patch,
        other two patches are wedge_left and wedge_right. Sides are named
        according to this sketch:

                        outer
            _________________________________
            |                               |    
            | left                    right | 
            |_______________________________|
                        core
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
        raise NotImplementedError("Use set_[outer|core|left|right]_patch methods for wedges")

    def set_outer_patch(self, patch_name):
        super().set_patch('back', patch_name)
    
    def set_core_patch(self, patch_name):
        super().set_patch('front', patch_name)

    def set_left_patch(self, patch_name):
        super().set_patch('left', patch_name)

    def set_right_patch(self, patch_name):
        super().set_patch('right', patch_name)
    
