import numba  # type:ignore
import numpy as np
from nptyping import Int32, NDArray, Shape

from classy_blocks.cbtyping import NPPointListType, NPPointType, NPVectorType
from classy_blocks.util.constants import VSMALL

NPIndexType = NDArray[Shape["*, 1"], Int32]


@numba.jit(nopython=True)
def scale_quality(base: float, exponent: float, factor: float, value: float) -> float:
    return factor * base ** (exponent * value) - factor


@numba.jit(nopython=True)
def scale_aspect(ratio: float) -> float:
    return scale_quality(4, 3, 2, np.log10(ratio))


@numba.jit(nopython=True)
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
def get_quad_normal(points: NPPointListType) -> tuple[NPVectorType, NPVectorType, float]:
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
def scale_angle(angle: float) -> float:
    n = 6
    m = 100
    threshold = 75
    a = m / (n * threshold ** (n - 1))

    if angle <= threshold:
        return a * angle**n

    return a * threshold**n + m * (angle - threshold)


@numba.jit(nopython=True, cache=True)
def get_quad_non_ortho(points: NPPointListType, center: NPPointType, normal: NPVectorType, corner: int) -> float:
    this_point = points[corner]
    next_point = points[(corner + 1) % 4]

    # non-ortho angle: angle between side_normal and center-side_center
    side_center = (this_point + next_point) / 2
    side_vector = next_point - this_point
    side_normal = np.cross(normal, side_vector)
    side_normal /= np.linalg.norm(side_normal) + VSMALL

    center_vector = center - side_center
    center_vector /= np.linalg.norm(center_vector) + VSMALL

    # non-orthogonality angle covers values from 0 to +/-180 degrees
    # and values above +/-70-ish are unacceptable regardless of the orientation;
    # therefore it makes no sense checking for inverted/degenerate quads
    return 180 * np.arccos(np.dot(side_normal, center_vector)) / np.pi


@numba.jit(nopython=True, cache=True)
def get_quad_inner_angle(points: NPPointListType, normal: NPVectorType, corner: int) -> float:
    next_side = points[(corner + 1) % 4] - points[corner]
    next_side /= np.linalg.norm(next_side) + VSMALL
    prev_side = points[(corner - 1) % 4] - points[corner]
    prev_side /= np.linalg.norm(prev_side) + VSMALL

    inner_angle = 180 * np.arccos(np.dot(next_side, prev_side)) / np.pi
    # ranges from 0 to 360 degrees but arccos only covert 0...180;
    # use normal to check for the rest
    if np.dot(np.cross(next_side, prev_side), normal) < 0:
        inner_angle += 180

    return inner_angle


@numba.jit(nopython=True, cache=True)
def get_quad_quality(grid_points: NPPointListType, cell_indexes: NPIndexType) -> float:
    quality = 0
    quad_points = take(grid_points, cell_indexes)
    center, normal, aspect = get_quad_normal(quad_points)

    for i in range(4):
        non_ortho_angle = get_quad_non_ortho(quad_points, center, normal, i)
        quality += scale_angle(non_ortho_angle)

        inner_angle = get_quad_inner_angle(quad_points, normal, i) - 90
        quality += scale_angle(inner_angle)

    quality += scale_aspect(aspect)

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
        center_vector /= np.linalg.norm(center_vector) + VSMALL

        non_ortho_angle = 180 * np.arccos(min(1 - VSMALL, np.dot(side_normal, center_vector))) / np.pi
        quality += scale_angle(non_ortho_angle)

        # take inner angles and aspect from quad calculation;
        for i in range(4):
            inner_angle = get_quad_inner_angle(side_points, side_normal, i) - 90
            quality += scale_angle(inner_angle)

        quality += scale_aspect(side_aspect)

    return quality
