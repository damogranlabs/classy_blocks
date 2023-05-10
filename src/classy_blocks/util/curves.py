# stuff that works on lists of points

from typing import Literal

import numpy as np
import scipy.spatial

from classy_blocks.util import functions as f


def dilute_indexes(max_index_val: int, num_of_points: int) -> np.ndarray:
    """Generate <num_of_points> points from an array of length <max_index_val>."""
    return np.round(np.linspace(0, max_index_val - 1, num_of_points)).astype(int)


def dilute_points(points: np.ndarray, num_of_points: int) -> np.ndarray:
    """from an array of <points>, choose <L> equally-spaced points"""
    return points[dilute_indexes(len(points), num_of_points)]


def curve_length(points) -> float:
    """returns length of a curve given by <points>"""
    num = 0
    for i in range(len(points) - 1):
        num += scipy.spatial.distance.euclidean(points[i], points[i + 1])

    return num


def to_cartesian(points, direction: Literal[1, -1] = 1, rotation_axis: Literal["x", "z"] = "z") -> np.ndarray:
    """same as functions.to_cartesian but works on a list of points"""
    return np.array([f.to_cartesian(point, direction, rotation_axis) for point in points])
