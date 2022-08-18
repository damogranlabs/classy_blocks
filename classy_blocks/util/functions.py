# -*- coding: utf-8 -*-
import numpy as np

import scipy
import scipy.linalg
import scipy.optimize
import scipy.spatial

from ..util import constants as c

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
    point = np.asarray(point)
    axis = np.asarray(axis)
    origin = np.asarray(origin)

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

def arc_length_3point(A, B, C):
    """ Returns length of arc defined by 3 points, A, B and C; B is the point in between """
    ### Meticulously transcribed from 
    # https://develop.openfoam.com/Development/openfoam/-/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.C
    p1 = np.asarray(A)
    p2 = np.asarray(B)
    p3 = np.asarray(C)

    a = p2 - p1
    b = p3 - p1

    # Find centre of arcEdge
    asqr = a.dot(a)
    bsqr = b.dot(b)
    adotb = a.dot(b)

    denom = asqr*bsqr - adotb*adotb
    # if norm(denom) < 1e-5:
    #     raise ValueError("Invalid arc points!")

    fact = 0.5*(bsqr - adotb)/denom

    centre = p1 + 0.5*a + fact*(np.cross(np.cross(a, b), a))

    # Position vectors from centre
    r1 = p1 - centre
    r2 = p2 - centre
    r3 = p3 - centre

    mag1 = norm(r1)
    mag3 = norm(r3)

    # The radius from r1 and from r3 will be identical
    radius = r3

    # Determine the angle
    angle = np.arccos((r1.dot(r3))/(mag1*mag3))

    # Check if the vectors define an exterior or an interior arcEdge
    if np.dot(np.cross(r1, r2), np.cross(r1, r3)) < 0:
        angle = 2*np.pi - angle

    return angle*norm(radius)

def distance_from_line(line_point_1, line_point_2, p):
    """ Returns distance from point p from line, defined by two points """
    # TODO: TEST
    axis = line_point_2 - line_point_1
    vector = p - line_point_1

    return norm(np.cross(axis, vector))/norm(axis)
