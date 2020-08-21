# -*- coding: utf-8 -*-
import math, os

import numpy as np

import scipy
import scipy.linalg
import scipy.optimize
import scipy.spatial

def vector(x, y, z):
    """ A shortcut for creating 3D-space vectors;
    in case you need a lot of manual np.array([...]) """
    return np.array([x, y, z])

def deg2rad(deg):
    """ Convert degrees (input) to radians """
    return deg*np.pi/180.

def rad2deg(rad):
    """ convert radians (input) to degrees """
    return rad*180./np.pi

def norm(vector):
    """ a shortcut to scipy.linalg.norm() """
    return scipy.linalg.norm(vector)

def unit_vector(vector):
    """ Returns a vector of magnitude 1 with the same direction"""
    return vector / norm(vector)

def angle_between(v1, v2):
    """ Returns the angle between vectors 'v1' and 'v2', in radians:

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

def arbitrary_rotation_matrix(axis, theta):
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

    # Also Kudos to the guy with another answer for the same question (used here): """
    return scipy.linalg.expm(np.cross(np.eye(3), axis/norm(axis)*theta))
    
def arbitrary_rotation(point, axis, theta, origin):
    """ Rotate a point around any axis given by axis by angle theta [radians] """

    rotated_point = np.dot(arbitrary_rotation_matrix(axis, theta), point - origin)
    return rotated_point + origin

def rotate(point, angle, axis='x'):
    """ Rotate a point around a given axis by specified angle """
    if axis == 'y':
        axis = vector(0, 1, 0)
    elif axis == 'z':
        axis = vector(0, 0, 1)
    elif axis == 'x':
        axis = vector(1, 0, 0)
    else:
        raise ValueError("Rotation axis should be either 'x', 'y', or 'z' ")

    return arbitrary_rotation(point, axis, angle, vector(0, 0, 0))

def to_polar(point, axis='z'):
    """ Convert (x, y, z) point to (radius, angle, height);
    the axis of the new polar coordinate system can be chosen ('x' or 'z') """

    assert axis in ['x', 'z']

    if axis == 'z':
        radius = (point[0]**2 + point[1]**2)**0.5
        angle = np.arctan2(point[1], point[0])
        height = point[2]
    else: # axis == 'x'
        radius = (point[1]**2 + point[2]**2)**0.5
        angle = np.arctan2(point[2], point[1])
        height = point[0]

    return vector(radius, angle, height)

def to_cartesian(p, direction=1, axis='z'):
    """ Converts a point given in (r, theta, z) coordinates to
    cartesian coordinate system.

    optionally, axis can be aligned with either cartesian axis x* or z and
    rotation sense can be inverted with direction=-1

    *when axis is 'x': theta goes from 0 at y-axis toward z-axis

    """
    assert direction in [-1, 1]
    assert axis in ['x', 'z']

    radius = p[0]
    angle = direction*p[1]
    height = p[2]

    if axis == 'z':
        return vector(radius*np.cos(angle), radius*np.sin(angle), height)
        
    # axis == 'x'
    return vector( height, radius*np.cos(angle), radius*np.sin(angle) )

def lin_map(x, x_min, x_max, out_min, out_max, limit=False):
    """ map x that should take values from x_min to x_max
        to values out_min to out_max"""
    r = float(x - x_min) * float(out_max - out_min) / \
        float(x_max - x_min) + float(out_min)

    if limit:
        return sorted([out_min, r, out_max])[1]
    else:
        return r

def xy_line_intersection(p_1, p_2, p_3, p_4):
    """ p_1 and p_2 define the first line, p_3 and p_4 define the second; 
        return a point of intersection between these two lines in x-y plane

        Kudos: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
    """
    # only take x and y coordinates
    x1 = p_1[0]
    y1 = p_1[1]

    x2 = p_2[0]
    y2 = p_2[1]

    x3 = p_3[0]
    y3 = p_3[1]

    x4 = p_4[0]
    y4 = p_4[1]

    def det(p1, p2, p3, p4):
        return np.linalg.det(np.array([[p1, p2], [p3, p4]]))

    Dx1 = det(x1, y1, x2, y2)
    Dx2 = det(x1,  1, x2,  1)
    Dx3 = det(x3, y3, x4, y4)
    Dx4 = det(x3,  1, x4,  1)
    Dx5 = Dx2
    Dx6 = det(y1,  1, y2,  1)
    Dx7 = Dx4
    Dx8 = det(y3,  1, y4,  1)

    # x-coordinate
    Px = det(Dx1, Dx2, Dx3, Dx4)/det(Dx5, Dx6, Dx7, Dx8)

    # y-coordinate
    Dy1 = Dx1
    Dy2 = Dx6
    Dy3 = Dx3
    Dy4 = Dx8
    Dy5 = Dx2
    Dy6 = Dx6
    Dy7 = Dx7
    Dy8 = Dx8

    Py = det(Dy1, Dy2, Dy3, Dy4)/det(Dy5, Dy6, Dy7, Dy8)
    return vector(Px, Py, 0)

    # alternative solution with vectors
    # A = np.array([
    #     [p_2[0] - p_1[0], p_4[0] - p_3[0]],
    #     [p_2[1] - p_1[1], p_4[1] - p_3[1]],
    # ])
    # 
    # b = np.array([p_3[0] - p_1[0], p_3[1] - p_1[1]])
    # 
    # k1k2 = np.linalg.solve(A, b)
    # k1 = k1k2[0]
    # k2 = k1k2[1]
    # 
    # va = vector(
    #     p_1[0] + k1*(p_2[0] - p_1[0]),
    #     p_1[1] + k1*(p_2[1] - p_1[1]),
    #     0
    # )
    # 
    # vb = vector(
    #     p_3[0] + k2*(p_4[0] - p_3[0]),
    #     p_3[1] + k2*(p_4[1] - p_3[1]),
    #     0
    # )
    # 
    # print(P-va, P-vb, norm(va-vb))
    # return va

def extend_to_y(p_1, p_2, y):
    """ Return a point that lies on a line defined by p_1 and p_2 and on y=y; only in xy-plane! """
    fk_3 = lambda k: p_1[1] + k*(p_2 - p_1)[1] - y
    k_3 = scipy.optimize.newton(fk_3, 0)

    return p_1 + k_3*(p_2 - p_1)


