"""helper functions used by hexmesh"""

from typing import Iterable, Iterator

import numpy as np
from scipy.spatial import distance

from classy_blocks.types import NPPointListType, NPPointType, OrientType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


def ratio_to_vertex(x_ratio: float | None, y_ratio: float | None, z_ratio: float | None) -> int | None:
    """returns the vertex index for the point defined by the 3 ratios"""

    iv = None
    iv = 0 if (x_ratio == 0.0 and y_ratio == 0.0 and z_ratio == 0.0) else iv
    iv = 1 if (x_ratio == 1.0 and y_ratio == 0.0 and z_ratio == 0.0) else iv
    iv = 2 if (x_ratio == 1.0 and y_ratio == 1.0 and z_ratio == 0.0) else iv
    iv = 3 if (x_ratio == 0.0 and y_ratio == 1.0 and z_ratio == 0.0) else iv
    iv = 4 if (x_ratio == 0.0 and y_ratio == 0.0 and z_ratio == 1.0) else iv
    iv = 5 if (x_ratio == 1.0 and y_ratio == 0.0 and z_ratio == 1.0) else iv
    iv = 6 if (x_ratio == 1.0 and y_ratio == 1.0 and z_ratio == 1.0) else iv
    iv = 7 if (x_ratio == 0.0 and y_ratio == 1.0 and z_ratio == 1.0) else iv

    return iv


def ratio_to_wire(
    orient: OrientType | None, x_ratio: float | None, y_ratio: float | None, z_ratio: float | None
) -> int | None:
    """returns the wire index for the point defined by the 3 ratios"""

    iw: int | None = None
    if orient in [None, "bottom", "front"]:
        iw = 0 if (y_ratio == 0.0 and z_ratio == 0.0) else iw  # edge 0
    if orient in [None, "bottom", "back"]:
        iw = 1 if (y_ratio == 1.0 and z_ratio == 0.0) else iw  # edge 1
    if orient in [None, "top", "back"]:
        iw = 2 if (y_ratio == 1.0 and z_ratio == 1.0) else iw  # edge 2
    if orient in [None, "top", "front"]:
        iw = 3 if (y_ratio == 0.0 and z_ratio == 1.0) else iw  # edge 3
    if orient in [None, "bottom", "left"]:
        iw = 4 if (x_ratio == 0.0 and z_ratio == 0.0) else iw  # edge 4
    if orient in [None, "bottom", "right"]:
        iw = 5 if (x_ratio == 1.0 and z_ratio == 0.0) else iw  # edge 5
    if orient in [None, "top", "right"]:
        iw = 6 if (x_ratio == 1.0 and z_ratio == 1.0) else iw  # edge 6
    if orient in [None, "top", "left"]:
        iw = 7 if (x_ratio == 0.0 and z_ratio == 1.0) else iw  # edge 7
    if orient in [None, "front", "left"]:
        iw = 8 if (x_ratio == 0.0 and y_ratio == 0.0) else iw  # edge 8
    if orient in [None, "front", "right"]:
        iw = 9 if (x_ratio == 1.0 and y_ratio == 0.0) else iw  # edge 9
    if orient in [None, "back", "right"]:
        iw = 10 if (x_ratio == 1.0 and y_ratio == 1.0) else iw  # edge 10
    if orient in [None, "back", "left"]:
        iw = 11 if (x_ratio == 0.0 and y_ratio == 1.0) else iw  # edge 11

    return iw


def wire_to_ratio(iw: int, t_ratio: float) -> list[float | None]:
    """returns the 3 ratios based on wire index"""

    x_ratio: float | None = None
    y_ratio: float | None = None
    z_ratio: float | None = None
    if iw == 0:
        x_ratio = t_ratio
        y_ratio = 0.0
        z_ratio = 0.0
    elif iw == 1:
        x_ratio = t_ratio
        y_ratio = 1.0
        z_ratio = 0.0
    elif iw == 2:
        x_ratio = t_ratio
        y_ratio = 1.0
        z_ratio = 1.0
    elif iw == 3:
        x_ratio = t_ratio
        y_ratio = 0.0
        z_ratio = 1.0
    elif iw == 4:
        x_ratio = 0.0
        y_ratio = t_ratio
        z_ratio = 0.0
    elif iw == 5:
        x_ratio = 1.0
        y_ratio = t_ratio
        z_ratio = 0.0
    elif iw == 6:
        x_ratio = 1.0
        y_ratio = t_ratio
        z_ratio = 1.0
    elif iw == 7:
        x_ratio = 0.0
        y_ratio = t_ratio
        z_ratio = 1.0
    elif iw == 8:
        x_ratio = 0.0
        y_ratio = 0.0
        z_ratio = t_ratio
    elif iw == 9:
        x_ratio = 1.0
        y_ratio = 0.0
        z_ratio = t_ratio
    elif iw == 10:
        x_ratio = 1.0
        y_ratio = 1.0
        z_ratio = t_ratio
    elif iw == 11:
        x_ratio = 0.0
        y_ratio = 1.0
        z_ratio = t_ratio

    return [x_ratio, y_ratio, z_ratio]


def trilinear_interp(
    blk: list, x_ratio: float | None = 0, y_ratio: float | None = 0, z_ratio: float | None = 0
) -> NPPointType:
    """Performs trilinear interpolation to find point defined by the three ratios"""

    # using a trilinear interpolation to get the corners of the hex
    # using the simple alogrithim
    # https://paulbourke.net/miscellaneous/interpolation/
    # this does not allow for edge shape or projection this is added later
    if x_ratio is not None and y_ratio is not None and z_ratio is not None:
        return (
            blk[0] * (1 - x_ratio) * (1 - y_ratio) * (1 - z_ratio)
            + blk[1] * (x_ratio) * (1 - y_ratio) * (1 - z_ratio)
            + blk[3] * (1 - x_ratio) * (y_ratio) * (1 - z_ratio)
            + blk[4] * (1 - x_ratio) * (1 - y_ratio) * (z_ratio)
            + blk[5] * (x_ratio) * (1 - y_ratio) * (z_ratio)
            + blk[7] * (1 - x_ratio) * (y_ratio) * (z_ratio)
            + blk[2] * (x_ratio) * (y_ratio) * (1 - z_ratio)
            + blk[6] * (x_ratio) * (y_ratio) * (z_ratio)
        )
    # default to first vertex point
    return blk[0]


def traverse_list(the_list, level=0, list_types=(list, tuple)) -> Iterator[str]:
    """iterates through list or list of lists"""
    if isinstance(the_list, list_types):
        for each_item in the_list:
            level += 1
            if isinstance(each_item, Iterable) and not isinstance(each_item, (str)):
                for sub_item in traverse_list(each_item, level, list_types):
                    yield from sub_item
            else:
                yield each_item
    else:
        yield the_list


def remove_duplicate_points(points: NPPointListType) -> list[NPPointType]:
    """removes duplicate points from array of 3d points"""
    # note the change in datatype in this routine
    _points: list[NPPointType] = []
    if len(points) > 0:
        _points.append(points[0])
        for ip in range(1, len(points)):
            idx = distance.cdist([points[ip]], _points).argmin()
            if f.norm(_points[idx] - points[ip]) > constants.TOL:
                _points.append(points[ip])
    return _points


# def chunker(seq, size):
#    """return chunk of size size from seq."""
#    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def tratio_on_line(
    apos: NPPointType | None = None, bpos: NPPointType | None = None, cpos: NPPointType | None = None
) -> float | None:
    """find t ratio of normal to line a b that goes through c ."""

    t_ratio = 0.0
    if apos is None or bpos is None or cpos is None:
        return t_ratio

    ab = bpos - apos
    magab = f.norm(ab)
    # coincident a and b
    if magab == 0:
        return t_ratio

    ac = cpos - apos
    magac = f.norm(ac)
    # coincident a and c
    if magac == 0:
        return t_ratio

    angle_ac_ab = f.angle_between(ac, ab)
    magat = magac * np.cos(angle_ac_ab)

    t_ratio = magat / magab
    if t_ratio > 1.0:
        t_ratio = 1.0

    return t_ratio


def closest_point_on_plane(
    ray_point: NPPointType | None, plane_point: NPPointType | None, plane_normal: NPPointType | None
) -> NPPointType | None:
    """Returns the position on a plane defined by p1 nd n1 that is closest to point q1"""

    if ray_point is None or plane_point is None or plane_normal is None:
        return None

    p1p2 = ray_point - plane_point

    # coincident ray_point and plane_point
    if f.norm(p1p2) == 0:
        return ray_point

    try:
        n2_unit = f.unit_vector(plane_normal)
        dist_to_plane = np.dot(p1p2, n2_unit)
        p_normal = dist_to_plane * n2_unit
        p_tangent = p1p2 - p_normal
        closest_point = plane_point + p_tangent

    except Exception:
        closest_point = None

    return closest_point


def intersect_line_plane(
    ray_point: NPPointType | None,
    ray_direction: NPPointType | None,
    plane_point: NPPointType | None,
    plane_normal: NPPointType | None,
    epsilon=1e-6,
) -> NPPointType | None:
    """
    ray_point, ray_direction: Define the line.
    plane_point, planeDirection: define the plane:

    Return a Vector or None (when the intersection can't be found).
    """

    # https://stackoverflow.com/questions/5666222/3d-line-plane-intersection

    if ray_point is None or ray_direction is None or plane_point is None or plane_normal is None:
        return None

    n_dot_u = np.dot(plane_normal, ray_direction)

    if abs(n_dot_u) > epsilon:
        r_to_p = plane_point - ray_point
        tscale = np.dot(plane_normal, r_to_p) / n_dot_u
        int_on_plane = ray_point + tscale * ray_direction

        return int_on_plane

    # The segment is parallel to plane.
    return None


def intersect_line_triangle(
    ray_point: NPPointType | None,
    ray_direction: NPPointType | None,
    p1: NPPointType | None,
    p2: NPPointType | None,
    p3: NPPointType | None,
) -> NPPointType | None:
    """Returns intersection of line and triangle"""

    # code from here
    # https://stackoverflow.com/questions/42740765/intersection-between-line-and-triangle-in-3d
    # and was created from maths found here
    # https://en.wikipedia.org/wiki/M%C3%B6ller%E2%80%93Trumbore_intersection_algorithm
    #
    # ray_point is the point from which the vector projects
    # ray_direction is the vector along which the projection is made
    # p1, p2, p3 are the ordinates of the triangle
    #
    def signed_tetra_volume(a, b, c, d):
        return np.sign(np.dot(np.cross(b - a, c - a), d - a) / 6.0)

    if ray_point is None or ray_direction is None or p1 is None or p2 is None or p3 is None:
        return None

    ray_point_p1 = ray_point + 1000.0 * ray_direction
    ray_point_m1 = ray_point - 1000.0 * ray_direction

    s1 = signed_tetra_volume(ray_point_m1, p1, p2, p3)
    s2 = signed_tetra_volume(ray_point_p1, p1, p2, p3)

    if s1 != s2:
        s3 = signed_tetra_volume(ray_point_m1, ray_point_p1, p1, p2)
        s4 = signed_tetra_volume(ray_point_m1, ray_point_p1, p2, p3)
        s5 = signed_tetra_volume(ray_point_m1, ray_point_p1, p3, p1)
        if s3 == s4 and s4 == s5:
            n = np.cross(p2 - p1, p3 - p1)
            t = np.dot(p1 - ray_point, n) / np.dot(ray_direction, n)
            return ray_point + t * ray_direction

    return None


def closest_triangle_point(
    ray_point: NPPointType | None, p1: NPPointType | None, p2: NPPointType | None, p3: NPPointType | None
) -> NPPointType | None:
    """Returns the closest triangle point to point ray_point"""
    # https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
    # ray_point is the point from which we are searching
    # p1, p2, p3 are the ordinates of the triangle

    if ray_point is None or p1 is None or p2 is None or p3 is None:
        return None

    ab = p2 - p1
    ac = p3 - p1
    ap = ray_point - p1

    d1 = np.dot(ab, ap)
    d2 = np.dot(ac, ap)

    if d1 <= 0 and d2 <= 0:
        return p1  # 1

    bp = ray_point - p2
    d3 = np.dot(ab, bp)
    d4 = np.dot(ac, bp)
    if d3 >= 0 and d4 <= d3:
        return p2  # 2

    cp = ray_point - p3
    d5 = np.dot(ab, cp)
    d6 = np.dot(ac, cp)
    if d6 >= 0 and d5 <= d6:
        return p3  # 3

    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return p1 + v * ab  # 4

    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        v = d2 / (d2 - d6)
        return p1 + v * ac  # 5

    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        v = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return p2 + v * (p3 - p2)  # 6

    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    return p1 + v * ab + w * ac  # 0


def closest_triangle_edge(
    ray_point: NPPointType | None, p1: NPPointType | None, p2: NPPointType | None, p3: NPPointType | None
) -> float | None:
    """Returns the closest point on a triangle edge to the ray_point"""

    # ray_point is the point from which we are searching
    # p1, p2, p3 are the ordinates of the triangle

    if ray_point is None or p1 is None or p2 is None or p3 is None:
        return None

    p12 = tratio_on_line(p1, p2, ray_point)
    p23 = tratio_on_line(p2, p3, ray_point)
    p31 = tratio_on_line(p3, p1, ray_point)

    d1 = f.norm(p12 - ray_point)
    d2 = f.norm(p23 - ray_point)
    d3 = f.norm(p31 - ray_point)

    rtn = None
    if d1 < d2:
        if d1 < d3:
            rtn = p12
        else:
            rtn = p31
    else:
        if d2 < d3:
            rtn = p23
        else:
            rtn = p31

    return rtn


def closest_triangle_vertex(
    ray_point: NPPointType | None, p1: NPPointType | None, p2: NPPointType | None, p3: NPPointType | None
) -> NPPointType | None:
    """Returns the closest triangle point to point q1"""

    # ray_point is the point from which we are searching
    # p1, p2, p3 are the ordinates of the triangle

    if ray_point is None or p1 is None or p2 is None or p3 is None:
        return None

    d1 = f.norm(p1 - ray_point)
    d2 = f.norm(p2 - ray_point)
    d3 = f.norm(p3 - ray_point)

    if d1 < d2:
        if d1 < d3:
            return p1
        else:
            return p3
    else:
        if d2 < d3:
            return p2
        else:
            return p3
    return None


def norm(matrix: NPPointType | None) -> float:
    # this is a fudge !!!
    if matrix is not None:
        return f.norm(matrix)

    return -1.0


def numbers_in_string(input_string: str) -> list:
    """finds numbers in a string"""
    numbers = []
    for t in input_string.replace("(", " ").replace(")", " ").split():
        try:
            numbers.append(float(t))
        except ValueError:
            pass
    return numbers
