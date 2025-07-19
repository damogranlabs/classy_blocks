from typing import TypeVar

import numpy as np
import scipy.spatial

from classy_blocks.base.exceptions import VertexNotFoundError
from classy_blocks.cbtyping import IndexType, NPPointListType, PointType
from classy_blocks.construct.flat.sketch import Sketch
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.util import functions as f
from classy_blocks.util.constants import TOL, vector_format

PointRegistryType = TypeVar("PointRegistryType", bound="PointRegistryBase")


class PointRegistryBase:
    """Searches for, connects and creates unique points, taken from lists of elements (quads, hexas, ...)"""

    cell_size: int  # 4 for quads, 8 for hexas

    def __init__(self, flattened_points: NPPointListType, merge_tol: float) -> None:
        if len(flattened_points) % self.cell_size != 0:
            raise ValueError(f"Number of points not divisible by cell_size: {len(flattened_points)} % {self.cell_size}")

        self.merge_tol = merge_tol

        # a flattened list of all points, possibly multiple at the same spot;
        # each 'cell' gets its own 'drawer' with indexes i...i+cell_size
        self._repeated_points = flattened_points
        self._repeated_point_tree = scipy.spatial.KDTree(self._repeated_points)

        # a list of unique points, analogous to blockMesh's vertex list
        self.unique_points = self._compile_unique()
        self._unique_point_tree = scipy.spatial.KDTree(self.unique_points)

        self.cell_addressing = [self._query_unique(self.find_cell_points(i)) for i in range(self.cell_count)]

    def _compile_unique(self) -> NPPointListType:
        # create a list of unique vertices, taken from the list of operations
        unique_points = []
        handled_indexes: set[int] = set()

        for point in self._repeated_points:
            coincident_indexes = self._repeated_point_tree.query_ball_point(point, r=self.merge_tol, workers=1)

            if set(coincident_indexes).isdisjoint(handled_indexes):
                # this vertex hasn't been handled yet
                unique_points.append(point)
                handled_indexes.update(coincident_indexes)

        return np.array(unique_points)

    def _query_unique(self, positions) -> list[int]:
        """A shortcut to KDTree.query_ball_point()"""
        result = self._unique_point_tree.query_ball_point(positions, r=self.merge_tol, workers=1)

        if len(np.shape(positions)) > 1:
            return f.flatten_2d_list(result)

        return result

    def _query_repeated(self, positions) -> list[int]:
        result = self._repeated_point_tree.query_ball_point(positions, r=self.merge_tol, workers=1)

        if len(np.shape(positions)) > 1:
            return f.flatten_2d_list(result)

        return result

    def find_point_index(self, position: PointType) -> int:
        """Returns the vertex at given position; raises an exception if multiple or none were found"""
        indexes = self._query_unique(position)
        if len(indexes) == 0:
            raise VertexNotFoundError(f"Vertex at {vector_format(position)} not found!")

        return indexes[0]

    def find_point_cells(self, position: PointType) -> list[int]:
        """Returns indexes of every cell that has a point at given position"""
        indexes = [i // self.cell_size for i in self._query_repeated(position)]

        return list(set(indexes))

    def find_cell_points(self, cell: int) -> NPPointListType:
        """Returns points that define this cell"""
        start_index = cell * self.cell_size
        end_index = start_index + self.cell_size
        return self._repeated_points[start_index:end_index]

    def find_cell_indexes(self, cell: int) -> list[int]:
        """Returns indexes of points that define this cell"""
        return self.cell_addressing[cell]

    def find_cell_neighbours(self, cell: int) -> list[int]:
        """Returns indexes of this and every touching cell"""
        cell_points = self.find_cell_points(cell)

        indexes = []

        for point in cell_points:
            indexes += self.find_point_cells(point)

        return list(set(indexes))

    @staticmethod
    def flatten(points, length) -> NPPointListType:
        return np.reshape(points, (length, 3))

    @classmethod
    def from_addresses(
        cls: type[PointRegistryType], points: NPPointListType, addressing: list[IndexType], merge_tol: float = TOL
    ) -> PointRegistryType:
        all_points = cls.flatten(
            [np.take(points, addr, axis=0) for addr in addressing], len(addressing) * cls.cell_size
        )

        return cls(all_points, merge_tol)

    @property
    def cell_count(self) -> int:
        return len(self._repeated_points) // self.cell_size

    @property
    def point_count(self) -> int:
        return len(self.unique_points)


class QuadPointRegistry(PointRegistryBase):
    """A registry of points, taken from a list of quads"""

    cell_size = 4

    @classmethod
    def from_sketch(cls: type[PointRegistryType], sketch: Sketch, merge_tol: float = TOL) -> PointRegistryType:
        return cls(
            cls.flatten([face.point_array for face in sketch.faces], len(sketch.faces) * cls.cell_size), merge_tol
        )


class HexPointRegistry(PointRegistryBase):
    """A registry of points, taken from a list of hexas"""

    cell_size = 8

    @classmethod
    def from_operations(
        cls: type[PointRegistryType], operations: list[Operation], merge_tol: float = TOL
    ) -> PointRegistryType:
        all_points = cls.flatten([op.point_array for op in operations], len(operations) * cls.cell_size)

        return cls(all_points, merge_tol)
