import abc
import warnings
from typing import ClassVar, Dict, List, Optional, Set, Tuple

import numpy as np

from classy_blocks.base.exceptions import NoCommonSidesError
from classy_blocks.cbtyping import FloatListType, IndexType, NPPointListType, NPPointType, OrientType
from classy_blocks.optimize.connection import CellConnection
from classy_blocks.util import functions as f
from classy_blocks.util.constants import EDGE_PAIRS, VSMALL


class CellBase(abc.ABC):
    side_names: ClassVar[List[OrientType]]
    side_indexes: ClassVar[List[IndexType]]
    edge_pairs: ClassVar[List[Tuple[int, int]]]

    def __init__(self, grid_points: NPPointListType, indexes: IndexType):
        self.grid_points = grid_points
        self.indexes = indexes

        self.neighbours: Dict[OrientType, Optional[CellBase]] = {name: None for name in self.side_names}
        self.connections = [CellConnection(set(pair), {indexes[pair[0]], indexes[pair[1]]}) for pair in self.edge_pairs]

        # cache
        self._quality: Optional[float] = None

    def get_common_indexes(self, candidate: "CellBase") -> Set[int]:
        """Returns indexes of common vertices between this and provided cell"""
        this_indexes = set(self.indexes)
        cnd_indexes = set(candidate.indexes)

        return this_indexes.intersection(cnd_indexes)

    def get_corner(self, index: int) -> int:
        """Converts vertex index to local index
        (position of this vertex in the list)"""
        return self.indexes.index(index)

    def get_common_side(self, candidate: "CellBase") -> OrientType:
        """Returns orient of this cell that is shared with candidate"""
        common_vertices = self.get_common_indexes(candidate)

        if len(common_vertices) != len(self.side_indexes[0]):
            raise NoCommonSidesError

        corners = {self.get_corner(i) for i in common_vertices}

        for i, indexes in enumerate(self.side_indexes):
            if set(indexes) == corners:
                return self.side_names[i]

        raise NoCommonSidesError

    def add_neighbour(self, candidate: "CellBase") -> bool:
        """Adds the provided block to appropriate
        location in self.neighbours and returns True if
        this and provided block share a face;
        does nothing and returns False otherwise"""

        if candidate == self:
            return False

        try:
            orient = self.get_common_side(candidate)
            self.neighbours[orient] = candidate
            return True
        except NoCommonSidesError:
            return False

    @property
    def boundary(self) -> Set[int]:
        """Returns a list of indexes that define sides on boundary"""
        boundary = set()

        for i, side_name in enumerate(self.side_names):
            side_indexes = self.side_indexes[i]

            if self.neighbours[side_name] is None:
                boundary.update({self.indexes[si] for si in side_indexes})

        return boundary

    @property
    def points(self) -> NPPointListType:
        """A list of points defining this cell, as a numpy array"""
        return np.take(self.grid_points, self.indexes, axis=0)

    @property
    def center(self) -> NPPointType:
        """Center of this cell"""
        return np.average(self.points, axis=0)

    def get_side_points(self, i: int) -> NPPointListType:
        """In 2D, a 'side' is a line segment but in 3D it is a quadrangle"""
        return np.take(self.points, self.side_indexes[i], axis=0)

    def get_side_center(self, i: int) -> NPPointType:
        """Center point of each face (3D) or edge (2D)"""
        return np.average(self.get_side_points(i), axis=0)

    @abc.abstractmethod
    def get_side_normals(self, i: int) -> NPPointListType:
        """4 normals, each for every triangle of decomposed given face in 3D
        or a single normal of an edge in 2D"""

    @abc.abstractmethod
    def get_inner_angles(self, i: int) -> FloatListType:
        """Angles of 4 corners of given face (3D) or angle at
        the first corner of given edge (2D)"""

    def get_edge_lengths(self) -> FloatListType:
        points = self.points

        return np.array([f.norm(points[edge[1]] - points[edge[0]]) for edge in self.side_indexes])

    @property
    def quality(self):
        # quality calculation for a single cell, transcribed from
        # OpenFOAM checkMesh utility.
        # Chosen criteria are transformed with a rapidly increasing
        # function and summed up so that a single value is obtained
        # for optimization algorithm to minimize.
        # See documentation for in-depth explanation.

        # both 3D (cell) and 2d (face) use the same calculation but elements are different.

        quality = 0
        center = self.center

        def q_scale(base, exponent, factor, value):
            return factor * base ** (exponent * value) - factor

        try:
            warnings.filterwarnings("error")

            for orient, neighbour in self.neighbours.items():
                i = self.side_names.index(orient)

                ### non-orthogonality
                # angles between sides and self.center-neighbour.center or, if there's no neighbour
                # on this face, between face and self.center-face_center
                if neighbour is None:
                    # Cells at the wall simply use center of the face on the wall
                    # instead of their neighbour's center
                    c2c = center - self.get_side_center(i)
                else:
                    c2c = center - neighbour.center

                c2cn = c2c / np.linalg.norm(c2c)

                angles = 180 * np.arccos(np.dot(self.get_side_normals(i), c2cn)) / np.pi
                quality += np.sum(q_scale(1.25, 0.35, 0.8, angles))
                ### cell inner angles
                quality += np.sum(q_scale(1.5, 0.25, 0.15, abs(self.get_inner_angles(i))))

            ### aspect ratio: one number for the whole cell (not per side)
            edge_lengths = self.get_edge_lengths()
            side_max = max(edge_lengths)
            side_min = min(edge_lengths) + VSMALL
            aspect_factor = np.log10(side_max / side_min)

            quality += np.sum(q_scale(3, 2.5, 3, aspect_factor))

        except RuntimeWarning:
            raise ValueError(f"Degenerate Cell: {self}") from RuntimeWarning
        finally:
            warnings.resetwarnings()

        return quality

    @property
    def min_length(self) -> float:
        return min(self.get_edge_lengths())

    def __str__(self):
        return "-".join([str(index) for index in self.indexes])


class QuadCell(CellBase):
    # Like constants.FACE_MAP but for quadrangle sides as line segments
    side_names: ClassVar = ["front", "right", "back", "left"]
    side_indexes: ClassVar = [[0, 1], [1, 2], [2, 3], [3, 0]]
    edge_pairs: ClassVar = [(0, 1), (1, 2), (2, 3), (3, 0)]

    @property
    def normal(self):
        points = self.points

        return np.cross(points[1] - points[0], points[3] - points[0])

    def get_side_normals(self, i):
        side_points = self.get_side_points(i)
        side_vector = side_points[1] - side_points[0]

        normal = np.cross(self.normal, side_vector)

        return [f.unit_vector(normal)]

    def get_inner_angles(self, i):
        points = np.take(self.points, ((i - 1) % 4, i, (i + 1) % 4), axis=0)

        side_1 = f.unit_vector(points[2] - points[1])
        side_2 = f.unit_vector(points[0] - points[1])

        return np.expand_dims(180 * np.arccos(np.dot(side_1, side_2)) / np.pi - 90, axis=0)


class HexCell(CellBase):
    """A block, treated as a single cell;
    its quality metrics can then be transcribed directly
    from checkMesh."""

    # FACE_MAP, ordered and modified so that all faces point towards cell center;
    # provided their points are visited in an anti-clockwise manner
    # names and indexes must correspond (both must be in the same order)
    side_names: ClassVar = ["bottom", "top", "left", "right", "front", "back"]
    side_indexes: ClassVar = [[0, 1, 2, 3], [7, 6, 5, 4], [4, 0, 3, 7], [6, 2, 1, 5], [0, 4, 5, 1], [7, 3, 2, 6]]
    edge_pairs: ClassVar = EDGE_PAIRS

    def get_side_normals(self, i: int):
        side_center = self.get_side_center(i)
        side_points = self.get_side_points(i)

        side_1 = side_points - side_center
        side_2 = np.roll(side_points, -1, axis=0) - side_center

        side_normals = np.cross(side_1, side_2)

        nnorms = np.linalg.norm(side_normals, axis=1) + VSMALL
        return side_normals / nnorms[:, np.newaxis]

    def get_inner_angles(self, i: int):
        side_points = self.get_side_points(i)

        sides_1 = np.roll(side_points, -1, axis=0) - side_points
        side_1_norms = np.linalg.norm(sides_1, axis=1) + VSMALL
        sides_1 = sides_1 / side_1_norms[:, np.newaxis]

        sides_2 = np.roll(side_points, 1, axis=0) - side_points
        side_2_norms = np.linalg.norm(sides_2, axis=1) + VSMALL
        sides_2 = sides_2 / side_2_norms[:, np.newaxis]

        angles = np.sum(sides_1 * sides_2, axis=1)
        return 180 * np.arccos(angles) / np.pi - 90
