from collections import OrderedDict
from typing import Dict, List, Optional, Set

import numpy as np

from classy_blocks.items.block import Block
from classy_blocks.items.vertex import Vertex
from classy_blocks.types import NPPointListType, NPPointType, OrientType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import EDGE_PAIRS, FACE_MAP, VSMALL


class NoCommonSidesError(Exception):
    """Raised when two cells don't share a side"""


class Cell:
    """A block, treated as a single cell;
    its quality metrics can then be transcribed directly
    from checkMesh."""

    def __init__(self, block: Block):
        self.block = block
        self.neighbours: Dict[OrientType, Optional[Cell]] = {
            "bottom": None,
            "top": None,
            "left": None,
            "right": None,
            "front": None,
            "back": None,
        }

        self.vertex_indexes = [self.block.vertices[i].index for i in range(8)]

        # FACE_MAP, ordered and modified so that all faces point towards cell center;
        # provided their points are visited in an anti-clockwise manner
        q_map = OrderedDict()
        q_map["bottom"] = (0, 1, 2, 3)
        q_map["top"] = (7, 6, 5, 4)
        q_map["left"] = (4, 0, 3, 7)
        q_map["right"] = (6, 2, 1, 5)
        q_map["front"] = (0, 4, 5, 1)
        q_map["back"] = (7, 3, 2, 6)
        self.q_map = q_map
        self.side_indexes = [item[0] for item in q_map.items()]
        self.face_indexes = [item[1] for item in q_map.items()]

    def get_common_vertices(self, candidate: "Cell") -> Set[int]:
        """Returns indexes of common vertices between this and provided cell"""
        this_indexes = set(self.vertex_indexes)
        cnd_indexes = {vertex.index for vertex in candidate.block.vertices}

        return this_indexes.intersection(cnd_indexes)

    def get_corner(self, vertex_index: int) -> int:
        """Converts vertex index to local index
        (position of this vertex in the list)"""
        return self.vertex_indexes.index(vertex_index)

    def get_common_side(self, candidate: "Cell") -> OrientType:
        """Returns orient of this cell that is shared with candidate"""
        common_vertices = self.get_common_vertices(candidate)

        if len(common_vertices) != 4:
            raise NoCommonSidesError

        corners = {self.get_corner(i) for i in common_vertices}

        for orient, indexes in FACE_MAP.items():
            if set(indexes) == corners:
                return orient

        raise NoCommonSidesError

    def add_neighbour(self, candidate: "Cell") -> bool:
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
    def vertices(self) -> List[Vertex]:
        return self.block.vertices

    @property
    def points(self) -> NPPointListType:
        """A list of points defining this cell, as a numpy array"""
        return np.asarray([vertex.position for vertex in self.vertices])

    @property
    def center(self) -> NPPointType:
        """Center of this cell"""
        return np.average(self.points, axis=0)

    @property
    def face_points(self) -> NPPointListType:
        return np.take(self.points, self.face_indexes, axis=0)

    @property
    def face_centers(self) -> NPPointListType:
        """Center point of each face"""
        return np.average(self.face_points, axis=1)

    @property
    def quality(self) -> float:
        quality = 0

        center = self.center

        def q_scale(base, exponent, factor, value):
            return factor * base ** (exponent * value) - factor

        for orient, neighbour in self.neighbours.items():
            i = self.side_indexes.index(orient)
            # quality calculation for a single cell, transcribed from
            # OpenFOAM checkMesh utility.
            # Chosen criteria are transformed with a rapidly increasing
            # function and summed up so that a single value is obtained
            # for optimization algorithm to minimize.
            # See documentation for in-depth explanation.

            ### non-orthogonality
            # angles between faces and self.center-neighbour.center or, if there's no neighbour
            # on this face, between face and self.center-face_center
            face_points = self.face_points[i]
            face_center = self.face_centers[i]

            side_1 = face_points - face_center
            side_2 = np.roll(face_points, -1, axis=0) - face_center
            face_normals = np.cross(side_1, side_2)

            if neighbour is None:
                # Cells at the wall simply use center of the face on the wall
                # instead of their neighbour's center
                c2c = center - face_center
            else:
                c2c = center - neighbour.center

            c2cn = c2c / np.linalg.norm(c2c)

            nnorms = np.linalg.norm(face_normals, axis=1) + VSMALL
            normals = face_normals / nnorms[:, np.newaxis]

            angles = 180 * np.arccos(np.dot(normals, c2cn)) / np.pi

            quality += np.sum(q_scale(1.25, 0.35, 0.8, angles))

            ### cell inner angles
            sides_1 = np.roll(face_points, -1, axis=0) - face_points
            side_1_norms = np.linalg.norm(sides_1, axis=1) + VSMALL
            sides_1 = sides_1 / side_1_norms[:, np.newaxis]

            sides_2 = np.roll(face_points, 1, axis=0) - face_points
            side_2_norms = np.linalg.norm(sides_2, axis=1) + VSMALL
            sides_2 = sides_2 / side_2_norms[:, np.newaxis]

            angles = np.sum(sides_1 * sides_2, axis=1)
            angles = 180 * np.arccos(angles) / np.pi - 90

            quality += np.sum(q_scale(1.5, 0.25, 0.15, abs(angles)))

            ### aspect ratio
            side_max = max(side_1_norms)
            side_min = min(side_1_norms) + VSMALL
            aspect_factor = np.log10(side_max / side_min)

            quality += np.sum(q_scale(3, 2.5, 3, aspect_factor))

        return quality

    @property
    def reference_size(self) -> float:
        """Returns the length of the shortest edge;
        used for initial optimization step"""
        lengths = [f.norm(self.points[pair[0]] - self.points[pair[1]]) for pair in EDGE_PAIRS]

        return min(lengths)
