from typing import Dict, List, Optional, Set

import numpy as np

from classy_blocks.construct.flat.quad import Quad
from classy_blocks.types import NPPointListType, QuadIndexType
from classy_blocks.util import functions as f


class QuadMap:
    def __init__(self, positions: NPPointListType, indexes: List[QuadIndexType]):
        self.indexes = indexes

        self.quads = [Quad(positions, quad_indexes) for quad_indexes in indexes]

    def update(self, positions: Optional[NPPointListType] = None) -> None:
        if positions is None:
            positions = self.positions

        for quad in self.quads:
            quad.update(positions)

    @property
    def positions(self) -> NPPointListType:
        """Reconstructs positions back from faces so they are always up-to-date,
        even after transforms"""
        indexes = list(np.array(self.indexes).flatten())
        max_index = max(indexes)
        all_points = f.flatten_2d_list([quad.face.point_array.tolist() for quad in self.quads])

        return np.array([all_points[indexes.index(i)] for i in range(max_index + 1)])

    @property
    def connections(self) -> List[Set[int]]:
        return f.flatten_2d_list([quad.connections for quad in self.quads])

    @property
    def neighbours(self) -> Dict[int, Set[int]]:
        """Returns a dictionary point:[neighbour points] as defined by quads"""
        length = int(max(np.array(self.indexes).ravel())) + 1

        neighbours: Dict[int, Set[int]] = {i: set() for i in range(length)}
        connections = self.connections

        for connection in connections:
            clist = list(connection)
            neighbours[clist[0]].add(clist[1])
            neighbours[clist[1]].add(clist[0])

        return neighbours

    @property
    def boundary_points(self) -> Set[int]:
        """Returns indexes of points that can be smoothed"""
        connections = self.connections
        fixed_points: Set[int] = set()

        for edge in connections:
            if connections.count(edge) == 1:
                fixed_points.update(edge)

        return fixed_points

    def get_nearby_quads(self, index: int) -> Set[int]:
        """Returns a list of quads that contain each movable point"""
        indexes = set()

        for i, quad in enumerate(self.quads):
            if index in quad.indexes:
                indexes.add(i)

        return indexes

    def smooth_laplacian(
        self,
        fix_points: Optional[Set[int]] = None,
    ) -> None:
        """Smooth the points using laplacian smoothing;
        each point is moved to the average of its neighbours"""
        if fix_points is None:
            fix_points = set()

        fixed_points = self.boundary_points.union(fix_points)
        positions = self.positions

        for point_index, point_neighbours in self.neighbours.items():
            if point_index in fixed_points:
                continue

            nei_positions = [positions[i] for i in point_neighbours]
            corner_center = np.average(nei_positions, axis=0)

            nei_quads = [self.quads[i] for i in self.get_nearby_quads(point_index)]
            quad_center = np.average([quad.center for quad in nei_quads], axis=0)

            # distances = np.array([f.norm(pos - center) for pos in nei_positions])
            # ratios = distances / np.average(distances)
            # weights = ratios

            positions[point_index] = (corner_center + quad_center) / 2

        self.update(positions)
