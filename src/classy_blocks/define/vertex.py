"""Vertex object defines a point in space and all operations that can be applied to it."""
from typing import List, Callable
from classy_blocks import types

import numpy as np
from numpy.typing import ArrayLike

from classy_blocks.util import functions as f
from classy_blocks.util import constants

def transform_points(points: List[List[float]], function: Callable) -> List[List[float]]:
    """A comprehensive shortcut"""
    return [function(p) for p in points]

class Vertex:
    """point with an index that's used in block and face definition
    and can output in OpenFOAM format"""

    def __init__(self, point):
        self.point = np.asarray(point)
        self.mesh_index = None  # will be changed in Mesh.write()

    def translate(self, displacement:ArrayLike):
        """Move this point by 'displacement' vector"""
        return self.__class__(self.point + np.asarray(displacement, dtype=float))

    def rotate(self, angle:float, axis:ArrayLike, origin:ArrayLike=None):
        """ Rotate this vertex around an arbitrary axis and origin """
        if origin is None:
            origin = [0, 0, 0]

        # returns a new, rotated Vertex
        return self.__class__(f.arbitrary_rotation(self.point, axis, angle, origin))

    def scale(self, ratio:float, origin:ArrayLike):
        """'Scales' this point's position around given origin."""
        return self.__class__(self.scale_point(self.point, ratio, origin))

    @staticmethod
    def scale_point(point:ArrayLike, ratio:float, origin:ArrayLike) -> ArrayLike:
        """Scale point's position around origin."""
        return origin + (point - origin)*ratio
