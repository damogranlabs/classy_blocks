from typing import Dict, List, Set

import numpy as np
import scipy.optimize

from classy_blocks.construct.flat.quad import Quad
from classy_blocks.types import NPPointListType, QuadIndexType
from classy_blocks.util import functions as f


class QuadMap:
    def __init__(self, positions: NPPointListType, indexes: List[QuadIndexType]):
        self.indexes = indexes
        self.positions = positions

        self.quads = [Quad(self.positions, quad_indexes) for quad_indexes in indexes]

    def update(self) -> None:
        for quad in self.quads:
            quad.update()

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
    def fixed_points(self) -> Set[int]:
        """Returns indexes of points that can be smoothed"""
        connections = self.connections
        fixed_points: Set[int] = set()

        for edge in connections:
            if connections.count(edge) == 1:
                fixed_points.update(edge)

        return fixed_points

    def smooth_laplacian(self, iterations: int = 5) -> None:
        """Smooth the points using laplacian smoothing;
        each point is moved to the average of its neighbours"""
        neighbours = self.neighbours
        fixed_points = self.fixed_points

        for _ in range(iterations):
            for point_index, point_neighbours in neighbours.items():
                if point_index in fixed_points:
                    continue

                nei_positions = [self.positions[i] for i in point_neighbours]

                self.positions[point_index] = np.average(nei_positions, axis=0)

        self.update()

    def get_nearby_quads(self) -> Dict[int, List[Quad]]:
        """Returns a list of quads that contain each movable point"""
        corner_points: Dict[int, List[Quad]] = {}

        fixed_points = self.fixed_points

        for i, point in enumerate(self.positions):
            if i in fixed_points:
                continue

            corner_points[i] = []

            for quad in self.quads:
                if quad.contains(point):
                    corner_points[i].append(quad)

        return corner_points

    def optimize_energy(self):
        """Replace quad edges by springs and moves their positions
        so that all springs are in the most relaxed state possible,
        minimizing the energy of the system"""
        # get quads that are defined by each point
        fixed_points = self.fixed_points

        e1 = self.quads[0].e1
        e2 = self.quads[0].e2

        def energy(i, x) -> float:
            e = 0

            self.positions[i] = x[0] * e1 + x[1] * e2

            for quad in self.quads:
                quad.update()
                e += quad.energy

            return e

        for i in range(len(self.positions)):
            if i in fixed_points:
                continue

            initial = [1, 1]
            scipy.optimize.minimize(lambda x, i=i: energy(i, x), initial)
