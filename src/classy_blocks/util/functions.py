"""Mathematical functions for general everyday household use"""

from typing import Literal, Optional, Union

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.spatial

from classy_blocks.types import NPPointType, NPVectorType, PointListType, PointType, VectorType
from classy_blocks.util import constants


def vector(x: float, y: float, z: float) -> NPVectorType:
    """A shortcut for creating 3D-space vectors;
    in case you need a lot of manual np.array([...])"""
    return np.array([x, y, z])


def deg2rad(deg: float) -> float:
    """Convert degrees (input) to radians"""
    return deg * np.pi / 180.0


def rad2deg(rad: float) -> float:
    """convert radians (input) to degrees"""
    return rad * 180.0 / np.pi


def norm(matrix: Union[PointType, PointListType]) -> float:
    """a shortcut to scipy.linalg.norm()"""
    # for arrays of vectors:
    # matrix = np.asarray(matrix, dtype=constants.DTYPE)
    # return scipy.linalg.norm(matrix, axis=len(np.shape(matrix))-1)

    return float(scipy.linalg.norm(matrix))


def unit_vector(vect: VectorType) -> NPVectorType:
    """Returns a vector of magnitude 1 with the same direction"""
    vect = np.asarray(vect, dtype=constants.DTYPE)
    return vect / norm(vect)


def angle_between(vect_1: VectorType, vect_2: VectorType) -> float:
    """Returns the angle between vectors in radians:

    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793

    Kudos: https://stackoverflow.com/questions/2827393/
    """
    v1_u = unit_vector(vect_1)
    v2_u = unit_vector(vect_2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rotation_matrix(axis: VectorType, theta: float):
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


def rotate(point: PointType, angle: float, axis: VectorType, origin: PointType) -> NPPointType:
    """Rotate a point around an axis@origin by a given angle [radians]"""
    point = np.asarray(point, dtype=constants.DTYPE)
    axis = np.asarray(axis, dtype=constants.DTYPE)
    origin = np.asarray(origin, dtype=constants.DTYPE)

    rotated_point = np.dot(rotation_matrix(axis, angle), point - origin)
    return rotated_point + origin


def scale(point: PointType, ratio: float, origin: Optional[PointType]) -> NPPointType:
    """Scales a point around origin by specified ratio;
    if not specified, origin is taken as [0, 0, 0]."""
    point = np.asarray(point, dtype=constants.DTYPE)
    origin = np.asarray(origin, dtype=constants.DTYPE)

    return origin + (point - origin) * ratio


def to_polar(point: PointType, axis: Literal["x", "z"] = "z") -> NPVectorType:
    """Convert (x, y, z) point to (radius, angle, height);
    the axis of the new polar coordinate system can be chosen ('x' or 'z')"""
    if axis not in ["x", "z"]:
        raise ValueError(f"`axis` must be 'x' or 'z', got {axis}")

    if axis == "z":
        radius = (point[0] ** 2 + point[1] ** 2) ** 0.5
        angle = np.arctan2(point[1], point[0])
        height = point[2]
    else:  # axis == 'x'
        radius = (point[1] ** 2 + point[2] ** 2) ** 0.5
        angle = np.arctan2(point[2], point[1])
        height = point[0]

    return vector(radius, angle, height)


def to_cartesian(point: PointType, direction: Literal[1, -1] = 1, axis: Literal["x", "z"] = "z") -> NPPointType:
    """Converts a point given in (r, theta, z) coordinates to
    cartesian coordinate system.

    optionally, axis can be aligned with either cartesian axis x* or z and
    rotation sense can be inverted with direction=-1

    *when axis is 'x': theta goes from 0 at y-axis toward z-axis
    """
    if direction not in [-1, 1]:
        raise ValueError(f"`direction` must be '-1' or '1', got {direction}")
    if axis not in ["x", "z"]:
        raise ValueError(f"`axis` must be 'x' or 'z', got {axis}")

    radius = point[0]
    angle = direction * point[1]
    height = point[2]

    if axis == "z":
        return vector(radius * np.cos(angle), radius * np.sin(angle), height)

    # axis == 'x'
    return vector(height, radius * np.cos(angle), radius * np.sin(angle))


def lin_map(x: float, x_min: float, x_max: float, out_min: float, out_max: float, limit: bool = False) -> float:
    """map x that should take values from x_min to x_max
    to values out_min to out_max"""
    r = float(x - x_min) * float(out_max - out_min) / float(x_max - x_min) + float(out_min)

    if limit:
        return sorted([out_min, r, out_max])[1]
    else:
        return r


def arc_length_3point(p_start: NPPointType, p_btw: NPPointType, p_end: NPPointType) -> float:
    """Returns length of arc defined by 3 points"""
    ### Meticulously transcribed from
    # https://develop.openfoam.com/Development/openfoam/-/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.C

    vect_a = p_btw - p_start
    vect_b = p_end - p_start

    # Find centre of arcEdge
    asqr = vect_a.dot(vect_a)
    bsqr = vect_b.dot(vect_b)
    adotb = vect_a.dot(vect_b)

    denom = asqr * bsqr - adotb * adotb
    # https://develop.openfoam.com/Development/openfoam/-/blob/master/src/OpenFOAM/primitives/Scalar/floatScalar/floatScalar.H
    if norm(denom) < 1e-18:
        raise ValueError("Invalid arc points!")

    fact = 0.5 * (bsqr - adotb) / denom

    centre = p_start + 0.5 * vect_a + fact * (np.cross(np.cross(vect_a, vect_b), vect_a))

    # Position vectors from centre
    rad_start = p_start - centre
    rad_btw = p_btw - centre
    rad_end = p_end - centre

    mag1 = norm(rad_start)
    mag3 = norm(rad_end)

    # The radius from r1 and from r3 will be identical
    radius = rad_end

    # Determine the angle
    angle = np.arccos((rad_start.dot(rad_end)) / (mag1 * mag3))

    # Check if the vectors define an exterior or an interior arcEdge
    if np.dot(np.cross(rad_start, rad_btw), np.cross(rad_start, rad_end)) < 0:
        angle = 2 * np.pi - angle

    return angle * norm(radius)


def arc_mid(axis: VectorType, center: PointType, radius: float, point_1: PointType, point_2: PointType) -> PointType:
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


def mirror(point: PointType, normal: VectorType, origin: PointType):
    """Mirror a point around a plane, given by a normal and origin"""
    # brainlessly copied from https://gamemath.com/book/matrixtransforms.html
    point = np.asarray(point)
    normal = unit_vector(normal)
    origin = np.asarray(origin)

    n_x = normal[0]
    n_y = normal[1]
    n_z = normal[2]

    matrix = np.array(
        [
            [
                1 - 2 * n_x**2,
                -2 * n_x * n_y,
                -2 * n_x * n_z,
            ],
            [
                -2 * n_x * n_y,
                1 - 2 * n_y**2,
                -2 * n_y * n_z,
            ],
            [
                -2 * n_x * n_z,
                -2 * n_y * n_z,
                1 - 2 * n_z**2,
            ],
        ]
    )

    point -= origin
    rotated = point.dot(matrix)
    rotated += origin

    return rotated


def is_point_on_plane(origin: PointType, normal: VectorType, point: PointType) -> float:
    """Calculated distance between a point and a plane, defined by origin and normal vector"""
    origin = np.asarray(origin)
    normal = unit_vector(normal)
    point = np.asarray(point)

    if norm(origin - point) < constants.TOL:
        return True

    return abs(np.dot(unit_vector(point - origin), normal)) < constants.TOL


def point_to_line_distance(origin: PointType, direction: VectorType, point: PointType) -> float:
    """Calculates distance from a line, defined by a point and normal, and an arbitrary point in 3D space"""
    origin = np.asarray(origin)
    point = np.asarray(point)
    direction = np.asarray(direction)

    return norm(np.cross(point - origin, direction)) / norm(direction)
