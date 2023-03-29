"""Mathematical functions for general everyday household use"""
from typing import Union, Optional, Literal

import numpy as np

import scipy
import scipy.linalg
import scipy.optimize
import scipy.spatial

from classy_blocks.types import PointType, VectorType, PointListType, \
    NPPointType, NPVectorType
from classy_blocks.util import constants

def vector(x:float, y:float, z:float) -> NPVectorType:
    """A shortcut for creating 3D-space vectors;
    in case you need a lot of manual np.array([...])"""
    return np.array([x, y, z])


def deg2rad(deg:float) -> float:
    """Convert degrees (input) to radians"""
    return deg * np.pi / 180.0

def rad2deg(rad:float) -> float:
    """convert radians (input) to degrees"""
    return rad * 180.0 / np.pi

def norm(matrix:Union[PointType, PointListType]) -> float:
    """ a shortcut to scipy.linalg.norm() """
    # for arrays of vectors:
    #matrix = np.asarray(matrix, dtype=constants.DTYPE)
    #return scipy.linalg.norm(matrix, axis=len(np.shape(matrix))-1)

    return float(scipy.linalg.norm(matrix))

def unit_vector(vect:VectorType) -> NPVectorType:
    """Returns a vector of magnitude 1 with the same direction"""
    vect = np.asarray(vect, dtype=constants.DTYPE)
    return  vect / norm(vect)

def angle_between(v1:VectorType, v2:VectorType) -> float:
    """Returns the angle between vectors 'v1' and 'v2', in radians:

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793

    Kudos: https://stackoverflow.com/questions/2827393/
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def rotation_matrix(axis:VectorType, theta:float):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    # Kudos to
    # https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    #import math
    #
    # axis = np.asarray(axis)
    # axis = axis / math.sqrt(np.dot(axis, axis))
    # a = math.cos(theta / 2.0)
    # b, c, d = -axis * math.sin(theta / 2.0)
    # aa, bb, cc, dd = a * a, b * b, c * c, d * d
    # bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    # return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
    #                  [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
    #                  [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    # Also Kudos to the guy with another answer for the same question (used here):"""
    axis = np.asarray(axis)
    return scipy.linalg.expm(np.cross(np.eye(3), axis / norm(axis) * theta))


def rotate(point:PointType, axis:VectorType, angle:float, origin:PointType) -> NPPointType:
    """Rotate a point around any axis given by axis by angle theta [radians]"""
    point = np.asarray(point)
    axis = np.asarray(axis)
    origin = np.asarray(origin)

    rotated_point = np.dot(rotation_matrix(axis, angle), point - origin)
    return rotated_point + origin

def scale(point:PointType, ratio:float, origin:Optional[PointType]=None) -> NPPointType:
        """Scales a point around origin by specified ratio;
        if not specified, origin is taken as [0, 0, 0]."""
        if origin is None:
            origin = vector(0, 0, 0)

        origin = np.asarray(origin, dtype=constants.DTYPE)

        return origin + (point - origin)*ratio

def to_polar(point:PointType, axis:Literal['x', 'y', 'z']='z') -> NPVectorType:
    """Convert (x, y, z) point to (radius, angle, height);
    the axis of the new polar coordinate system can be chosen ('x' or 'z')"""

    assert axis in ["x", "z"]

    if axis == "z":
        radius = (point[0] ** 2 + point[1] ** 2) ** 0.5
        angle = np.arctan2(point[1], point[0])
        height = point[2]
    else:  # axis == 'x'
        radius = (point[1] ** 2 + point[2] ** 2) ** 0.5
        angle = np.arctan2(point[2], point[1])
        height = point[0]

    return vector(radius, angle, height)

def to_cartesian(
        point:PointType,
        direction:Literal[1, -1]=1,
        axis:Literal['x', 'z']='z') -> NPPointType:
    """Converts a point given in (r, theta, z) coordinates to
    cartesian coordinate system.

    optionally, axis can be aligned with either cartesian axis x* or z and
    rotation sense can be inverted with direction=-1

    *when axis is 'x': theta goes from 0 at y-axis toward z-axis

    """
    assert direction in [-1, 1]
    assert axis in ["x", "z"]

    radius = point[0]
    angle = direction * point[1]
    height = point[2]

    if axis == "z":
        return vector(radius * np.cos(angle), radius * np.sin(angle), height)

    # axis == 'x'
    return vector(height, radius * np.cos(angle), radius * np.sin(angle))


def lin_map(x:float, x_min:float, x_max:float, out_min:float, out_max:float, limit:bool=False) -> float:
    """map x that should take values from x_min to x_max
    to values out_min to out_max"""
    r = float(x - x_min) * float(out_max - out_min) / float(x_max - x_min) + float(out_min)

    if limit:
        return sorted([out_min, r, out_max])[1]
    else:
        return r

def arc_length_3point(A:PointType, B:PointType, C:PointType) -> float:
    """Returns length of arc defined by 3 points, A, B and C; B is the point in between"""
    ### Meticulously transcribed from
    # https://develop.openfoam.com/Development/openfoam/-/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.C

    # FIXME: handle 'invalid values encountered'
    p1 = np.asarray(A, dtype=constants.DTYPE)
    p2 = np.asarray(B, dtype=constants.DTYPE)
    p3 = np.asarray(C, dtype=constants.DTYPE)

    a = p2 - p1
    b = p3 - p1

    # Find centre of arcEdge
    asqr = a.dot(a)
    bsqr = b.dot(b)
    adotb = a.dot(b)

    denom = asqr * bsqr - adotb * adotb
    # if norm(denom) < 1e-5:
    #     raise ValueError("Invalid arc points!")

    fact = 0.5 * (bsqr - adotb) / denom

    centre = p1 + 0.5 * a + fact * (np.cross(np.cross(a, b), a))

    # Position vectors from centre
    r1 = p1 - centre
    r2 = p2 - centre
    r3 = p3 - centre

    mag1 = norm(r1)
    mag3 = norm(r3)

    # The radius from r1 and from r3 will be identical
    radius = r3

    # Determine the angle
    angle = np.arccos((r1.dot(r3)) / (mag1 * mag3))

    # Check if the vectors define an exterior or an interior arcEdge
    if np.dot(np.cross(r1, r2), np.cross(r1, r3)) < 0:
        angle = 2 * np.pi - angle

    return angle * norm(radius)

def arc_mid(
    axis: VectorType,
    center: PointType,
    radius: float,
    point_1: PointType, point_2: PointType) -> PointType:
    """Returns the midpoint of the specified arc in 3D space"""
    # Kudos to this guy for his shrewd solution
    # https://math.stackexchange.com/questions/3717427
    axis = np.asarray(axis, dtype=constants.DTYPE)
    center = np.asarray(center, dtype=constants.DTYPE)
    point_1 = np.asarray(point_1, dtype=constants.DTYPE)
    point_2 = np.asarray(point_2, dtype=constants.DTYPE)

    sec = point_2 - point_1
    sec_ort = np.cross(sec, axis)

    return center + unit_vector(sec_ort) * radius
