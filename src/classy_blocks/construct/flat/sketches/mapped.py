from typing import Dict, List, Set, Tuple

import numpy as np

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.flat.sketches.sketch import Sketch
from classy_blocks.types import NPPointListType, NPPointType, PointListType
from classy_blocks.util import functions as f
from classy_blocks.util.constants import DTYPE

QuadType = Tuple[int, int, int, int]


class Quad:
    """A helper class for tracking positions-faces-indexes-neighbours-whatnot"""

    def __init__(self, positions: NPPointListType, indexes: Tuple[int, int, int, int]):
        self.indexes = indexes
        self.face = Face(np.take(positions, list(indexes), axis=0))

    @property
    def connections(self) -> List[Set[int]]:
        return [{self.indexes[i], self.indexes[(i + 1) % 4]} for i in range(4)]


def get_connections(quad: QuadType) -> List[Set[int]]:
    return [{quad[i], quad[(i + 1) % 4]} for i in range(4)]


def get_all_connections(quads) -> List[Set[int]]:
    return f.flatten_2d_list([get_connections(quad) for quad in quads])


def find_neighbours(quads: List[QuadType]) -> Dict[int, Set[int]]:
    """Returns a dictionary point:[neighbour points] as defined by quads"""
    length = int(max(np.array(quads, dtype=DTYPE).ravel())) + 1

    neighbours: Dict[int, Set[int]] = {i: set() for i in range(length)}
    connections = get_all_connections(quads)

    for connection in connections:
        clist = list(connection)
        neighbours[clist[0]].add(clist[1])
        neighbours[clist[1]].add(clist[0])

    return neighbours


def get_fixed_points(quads) -> Set[int]:
    """Returns indexes of points that can be smoothed"""
    connections = get_all_connections(quads)
    fixed_points: Set[int] = set()

    for edge in connections:
        if connections.count(edge) == 1:
            fixed_points.update(edge)

    return fixed_points


def smooth(positions, quads, iterations: int) -> NPPointListType:
    neighbours = find_neighbours(quads)
    fixed_points = get_fixed_points(quads)

    for _ in range(iterations):
        for point_index, point_neighbours in neighbours.items():
            if point_index in fixed_points:
                continue

            nei_positions = np.take(positions, list(point_neighbours), axis=0)
            positions[point_index] = np.average(nei_positions, axis=0)

    return positions


class MappedSketch(Sketch):
    """A sketch that is created from predefined points.
    The points are connected to form quads which define Faces.
    There is only a single grid 'tier' in these sketches (faces = grid[0])."""

    def __init__(self, positions: PointListType, quads: List[Tuple[int, int, int, int]], smooth_iter: int = 0):
        positions = np.array(positions, dtype=DTYPE)

        if smooth_iter > 0:
            positions = smooth(positions, quads, smooth_iter)

        self.quads = [Quad(positions, quad) for quad in quads]

        self.add_edges()

    def add_edges(self) -> None:
        """An optional method that will add edges to wherever they need to be."""

    @property
    def faces(self):
        """A 'flattened' grid"""
        return [quad.face for quad in self.quads]

    @property
    def grid(self):
        return [self.faces]

    @property
    def center(self) -> NPPointType:
        """Center of this sketch"""
        return np.average([face.center for face in self.faces], axis=0)
