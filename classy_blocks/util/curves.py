# -*- coding: utf-8 -*-
# stuff that works on lists of points

import numpy as np
import scipy.spatial

from . import functions as f

def dilute_indexes(n, l):
    """ choose <l> points from an array of length <n> """
    return np.round(np.linspace(0, n-1, l)).astype(int)

def dilute_points(points, L):
    """ from an array of <points>, choose <L> equally-spaced points """
    return points[dilute_indexes(len(points), L)]

def curve_length(points):
    """ returns length of a curve given by <points> """
    l = 0
    for i in range(len(points) - 1):
        l += scipy.spatial.distance.euclidean(points[i], points[i+1])

    return l

def to_cartesian(points, direction=1, rotation_axis='z'):
    """ same as functions.to_cartesian but works on a list of points """
    return np.array(
        [f.to_cartesian(point, direction, rotation_axis)
            for point in points
        ])