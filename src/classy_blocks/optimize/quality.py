from typing import Tuple

import numba  # type:ignore
import numpy as np
from nptyping import Int32, NDArray, Shape

from classy_blocks.cbtyping import NPPointListType, NPPointType, NPVectorType
from classy_blocks.util.constants import VSMALL

NPIndexType = NDArray[Shape["*, 1"], Int32]


@numba.jit(nopython=True, cache=True)
def scale_quality(base: float, exponent: float, factor: float, value: float) -> float:
    return factor * base ** (exponent * value) - factor


@numba.jit(nopython=True, cache=True)
def scale_non_ortho(angle: float) -> float:
    return scale_quality(1.4, 0.25, 0.5, angle)


@numba.jit(nopython=True, cache=True)
def scale_inner_angle(angle: float) -> float:
    return scale_quality(1.4, 0.25, 0.5, np.abs(angle))


@numba.jit(nopython=True, cache=True)
def scale_aspect(ratio: float) -> float:
    return scale_quality(4, 3, 3, np.log10(ratio))


@numba.jit(nopython=True, cache=True)
def take(points: NPPointListType, indexes: NPIndexType):
    n_points = len(indexes)
    dim = points.shape[1]
    result = np.empty((n_points, dim), dtype=points.dtype)

    for i in range(n_points):
        for j in range(dim):
            result[i, j] = points[indexes[i], j]

    return result


@numba.jit(nopython=True, cache=True)
def get_center_point(points: NPPointListType) -> NPPointType:
    return np.sum(points, axis=0) / len(points)


@numba.jit(nopython=True, cache=True)
def get_quad_normal(points: NPPointListType) -> Tuple[NPVectorType, NPVectorType, float]:
    normal = np.zeros(3)
    center = get_center_point(points)

    min_length: float = 1e30
    max_length: float = 0

    for i in range(4):
        side_1 = points[i] - center
        side_2 = points[(i + 1) % 4] - center

        length = float(np.linalg.norm(side_2 - side_1))

        max_length = max(max_length, length)
        min_length = min(min_length, length)

        tri_normal = np.cross(side_1, side_2)
        tri_normal /= np.linalg.norm(tri_normal)

        normal += tri_normal

    return center, normal / np.linalg.norm(normal), max_length / (min_length + VSMALL)


@numba.jit(nopython=True, cache=True)
def is_quad_convex(points: NPPointListType) -> bool:
    # Compute normal using the first three points
    normal = np.cross(points[1] - points[0], points[2] - points[1])
    normal /= np.linalg.norm(normal)

    sign = 0
    for i in range(4):
        prev_leg = points[i] - points[(i - 1) % 4]
        next_leg = points[(i + 1) % 4] - points[i]
        cross = np.cross(prev_leg, next_leg)
        dot = np.dot(cross, normal)
        if i == 0:
            sign = np.sign(dot)
            if sign == 0:
                return False  # Degenerate
        else:
            if np.sign(dot) != sign:
                return False
    return True


@numba.jit(nopython=True, cache=True)
def scale_angle(angle: float) -> float:
    n = 4
    m = 10
    threshold = 65
    a = m / (n * threshold ** (n - 1))

    if angle <= threshold:
        return a * angle**n

    return a * threshold**n + m * (angle - threshold)


@numba.jit(nopython=True, cache=True)
def get_quad_non_ortho_quality(
    quad_points: NPPointListType, quad_center: NPPointType, quad_normal: NPPointType
) -> float:
    quality = 0

    for i in range(4):
        point_1 = quad_points[i]
        point_2 = quad_points[(i + 1) % 4]

        side_center = (point_1 + point_2) / 2
        side_vector = point_2 - point_1
        side_normal = np.cross(quad_normal, side_vector)
        side_normal /= np.linalg.norm(side_normal)

        center_vector = quad_center - side_center
        center_vector /= np.linalg.norm(center_vector)

        angle = 180 * np.arccos(np.dot(side_normal, center_vector)) / np.pi
        if not is_quad_convex(quad_points):
            angle += 180
        quality += scale_angle(angle)

    return quality


@numba.jit(nopython=True, cache=True)
def get_quad_angle_quality(quad_points: NPPointListType) -> float:
    quality = 0

    for i in range(4):
        corner_point = quad_points[i]
        next_point = quad_points[(i + 1) % 4]
        prev_point = quad_points[(i - 1) % 4]

        side_1 = next_point - corner_point
        side_1 /= np.linalg.norm(side_1) + VSMALL

        side_2 = prev_point - corner_point
        side_2 /= np.linalg.norm(side_2) + VSMALL

        angle = 180 * np.arccos(np.dot(side_1, side_2)) / np.pi - 90
        if not is_quad_convex(quad_points):
            angle += 180

        quality += scale_angle(angle)

    return quality


@numba.jit(nopython=True, cache=True)
def get_quad_quality(grid_points: NPPointListType, cell_indexes: NPIndexType) -> float:
    cell_points = take(grid_points, cell_indexes)
    cell_center, cell_normal, cell_aspect = get_quad_normal(cell_points)

    # non-ortho
    quality = get_quad_non_ortho_quality(cell_points, cell_center, cell_normal)

    # inner angles
    quality += get_quad_angle_quality(cell_points)

    # aspect ratio
    quality += scale_aspect(cell_aspect)

    return quality


@numba.jit(nopython=True, cache=True)
def get_hex_quality(grid_points: NPPointListType, cell_indexes: NPIndexType) -> float:
    cell_points = take(grid_points, cell_indexes)
    cell_center = get_center_point(cell_points)

    side_indexes = np.array([[0, 1, 2, 3], [7, 6, 5, 4], [4, 0, 3, 7], [6, 2, 1, 5], [0, 4, 5, 1], [7, 3, 2, 6]])

    quality = 0

    for side in side_indexes:
        # Non-ortho angle in a hexahedron is measured between two vectors:
        # <neighbour_center-this_center> and <face_normal>
        # but since cells on the boundary don't have a neighbour
        # simply <face_center> is taken.
        # For this kind of optimization it is quite sufficient to
        # take <face_center> for all cells.
        side_points = take(cell_points, side)
        side_center, side_normal, side_aspect = get_quad_normal(side_points)
        center_vector = cell_center - side_center

        center_vector /= np.linalg.norm(center_vector)

        angle = 180 * np.arccos(min(1 - VSMALL, np.dot(side_normal, center_vector))) / np.pi
        if not is_quad_convex(side_points):
            angle = 180 - angle
        quality += scale_angle(angle)

        # take inner angles and aspect from quad calculation;
        quality += get_quad_angle_quality(side_points)

        quality += scale_aspect(side_aspect)

    return quality
