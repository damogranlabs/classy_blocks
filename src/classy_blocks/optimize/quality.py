from typing import List, Tuple

import numba  # type:ignore
import numpy as np

from classy_blocks.cbtyping import NPPointListType, NPPointType, NPVectorType
from classy_blocks.util.constants import VSMALL


@numba.jit(nopython=True, cache=True)
def scale_quality(base: float, exponent: float, factor: float, value: float) -> float:
    return factor * base ** (exponent * value) - factor


@numba.jit(nopython=True, cache=True)
def scale_non_ortho(angle: float) -> float:
    return scale_quality(1.25, 0.35, 0.8, angle)


@numba.jit(nopython=True, cache=True)
def scale_inner_angle(angle: float) -> float:
    return scale_quality(1.5, 0.25, 0.15, angle)


@numba.jit(nopython=True, cache=True)
def scale_aspect(ratio: float) -> float:
    return scale_quality(3, 2.5, 3, np.log10(ratio))


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
        quality += scale_non_ortho(angle)

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
        quality += scale_inner_angle(angle)

    return quality


@numba.jit(nopython=True, cache=True)
def get_quad_quality(grid_points: NPPointListType, cell_indexes: List[int]) -> float:
    cell_points = np.take(grid_points, cell_indexes, axis=0)
    cell_center, cell_normal, cell_aspect = get_quad_normal(cell_points)

    # non-ortho
    quality = get_quad_non_ortho_quality(cell_points, cell_center, cell_normal)

    # inner angles
    quality += get_quad_angle_quality(cell_points)

    quality += scale_aspect(cell_aspect)

    return quality


@numba.jit(nopython=True, cache=True)
def get_hex_quality(grid_points: NPPointListType, cell_indexes: List[int]) -> float:
    cell_points = np.take(grid_points, cell_indexes, axis=0)
    cell_center = get_center_point(cell_points)

    side_indexes = np.array([[0, 1, 2, 3], [7, 6, 5, 4], [4, 0, 3, 7], [6, 2, 1, 5], [0, 4, 5, 1], [7, 3, 2, 6]])

    max_aspect = 1

    quality = 0

    for side in side_indexes:
        # Non-ortho angle is measured between two lines;
        # neighbour_center-to-this_center and face_center-to-this_center,
        # but for cells on the boundary simply face center is taken.
        # For this kind of optimization it is quite sufficient to take
        # only the latter as it's not much different and we'll optimize
        # other cells too.
        side_points = np.take(cell_points, side, axis=0)
        side_center, side_normal, side_aspect = get_quad_normal(side_points)
        center_vector = cell_center - side_center

        center_vector /= np.linalg.norm(center_vector)

        angle = 180 * np.arccos(np.dot(side_normal, center_vector)) / np.pi
        quality += scale_non_ortho(angle)

        max_aspect = max(max_aspect, side_aspect)

        # take inner angles and aspect from quad calculation
        quality += get_quad_angle_quality(side_points)

    quality += scale_aspect(max_aspect)

    return quality
