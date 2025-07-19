"""Mathematical functions for general everyday household use"""

from itertools import chain
from typing import Literal, Optional, Union

import numpy as np
import scipy
import scipy.linalg
from numba import jit  # type: ignore

from classy_blocks.cbtyping import NPPointListType, NPPointType, NPVectorType, PointListType, PointType, VectorType
from classy_blocks.util import constants


def vector(x: float, y: float, z: float) -> NPVectorType:
    """A shortcut for creating 3D-space vectors;
    in case you need a lot of manual np.array([...])"""
    return np.array([x, y, z], dtype=constants.DTYPE)


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


@jit(nopython=True, cache=True)
def _rotation_matrix(axis: NPVectorType, angle: float):
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    return np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
            [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc],
        ]
    )


def rotation_matrix(axis: VectorType, angle: float):
    """Returns the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians."""
    # Kudos to https://stackoverflow.com/questions/6802577/rotation-of-3d-vector
    axis = np.asarray(axis, dtype=constants.DTYPE)

    return _rotation_matrix(axis, angle)


@jit(nopython=True, cache=True)
def _rotate(point: NPPointType, angle: float, axis: NPVectorType, origin: NPPointType) -> NPPointType:
    # Rotation matrix associated with counterclockwise rotation about
    # the given axis by theta radians. Kudos to
    # https://stackoverflow.com/questions/6802577/rotation-of-3d-vector

    rotated_point = np.dot(_rotation_matrix(axis, angle), point - origin)
    return rotated_point + origin


def rotate(point: PointType, angle: float, axis: VectorType, origin: PointType) -> NPPointType:
    """Rotate a point around an axis@origin by a given angle [radians]"""
    point = np.asarray(point, dtype=constants.DTYPE)
    axis = np.asarray(axis, dtype=constants.DTYPE)
    origin = np.asarray(origin, dtype=constants.DTYPE)

    return _rotate(point, angle, axis, origin)


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


@jit(nopython=True, cache=True)
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
    if denom < 1e-18:
        raise ValueError("Invalid arc points!")

    fact = 0.5 * (bsqr - adotb) / denom

    centre = p_start + 0.5 * vect_a + fact * (np.cross(np.cross(vect_a, vect_b), vect_a))

    # Position vectors from centre
    rad_start = p_start - centre
    rad_btw = p_btw - centre
    rad_end = p_end - centre

    mag1 = np.linalg.norm(rad_start)
    mag3 = np.linalg.norm(rad_end)

    # The radius from r1 and from r3 will be identical
    radius = rad_end

    # Determine the angle
    angle = np.arccos((rad_start.dot(rad_end)) / (mag1 * mag3))

    # Check if the vectors define an exterior or an interior arcEdge
    if np.dot(np.cross(rad_start, rad_btw), np.cross(rad_start, rad_end)) < 0:
        angle = 2 * np.pi - angle

    return angle * np.linalg.norm(radius)


@jit(nopython=True, cache=True)
def divide_arc(center: NPPointType, point_1: NPPointType, point_2: NPPointType, count: int) -> NPPointListType:
    radius = np.linalg.norm(center - point_1)
    step = (point_2 - point_1) / (count + 1)
    result = np.empty((count, 3))

    for i in range(count):
        secant_point = point_1 + step * (i + 1)
        secant_vector = secant_point - center
        secant_length = np.linalg.norm(secant_vector)

        result[i] = center + radius * secant_vector / secant_length

    return result


@jit(nopython=True, cache=True)
def arc_mid(center: NPPointType, point_1: NPPointType, point_2: NPPointType) -> PointType:
    """Returns the midpoint of the specified arc in 3D space"""
    return divide_arc(center, point_1, point_2, 1)[0]


def mirror_matrix(normal: VectorType):
    n_x = normal[0]
    n_y = normal[1]
    n_z = normal[2]

    return np.array(
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


def mirror(point: PointType, normal: VectorType, origin: PointType):
    """Mirror a point around a plane, given by a normal and origin"""
    # brainlessly copied from https://gamemath.com/book/matrixtransforms.html
    point = np.asarray(point)
    normal = unit_vector(normal)
    origin = np.asarray(origin)

    point -= origin
    rotated = point.dot(mirror_matrix(normal))
    rotated += origin

    return rotated


def point_to_plane_distance(origin: PointType, normal: VectorType, point: PointType) -> float:
    origin = np.asarray(origin)
    normal = unit_vector(normal)
    point = np.asarray(point)

    if norm(origin - point) < constants.TOL:
        # point and origin are coincident
        return norm(origin - point)

    return abs(np.dot(point - origin, normal))


def is_point_on_plane(origin: PointType, normal: VectorType, point: PointType) -> bool:
    """Calculated distance between a point and a plane, defined by origin and normal vector"""
    return point_to_plane_distance(origin, normal, point) < constants.TOL


def point_to_line_distance(origin: PointType, direction: VectorType, point: PointType) -> float:
    """Calculates distance from a line, defined by a point and normal, and an arbitrary point in 3D space"""
    origin = np.asarray(origin)
    point = np.asarray(point)
    direction = np.asarray(direction)

    return norm(np.cross(point - origin, direction)) / norm(direction)


def polyline_length(points: NPPointListType) -> float:
    """Calculates length of a polyline, given by a list of points"""
    if len(np.shape(points)) != 2 or len(points[0]) != 3:
        raise ValueError("Provide a list of points in 3D space!")

    if np.shape(points)[0] < 2:
        raise ValueError("Use at least 2 points for a polyline!")

    return np.sum(np.sqrt(np.sum((points[:-1] - points[1:]) ** 2, axis=1)))


def flatten_2d_list(twodim: list[list]) -> list:
    """Flattens a list of lists to a 1d-list"""
    return list(chain.from_iterable(twodim))
