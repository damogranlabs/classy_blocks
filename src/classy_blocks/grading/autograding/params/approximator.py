import dataclasses
import itertools
from typing import List, Sequence

import numpy as np

from classy_blocks.grading.chop import Chop
from classy_blocks.types import FloatListType
from classy_blocks.util.constants import TOL


@dataclasses.dataclass
class Piece:
    coords: FloatListType

    @property
    def point_count(self) -> int:
        return len(self.coords)

    @property
    def cell_count(self) -> int:
        return self.point_count - 1

    @property
    def start_size(self) -> float:
        return abs(self.coords[1] - self.coords[0])

    @property
    def end_size(self) -> float:
        return abs(self.coords[-1] - self.coords[-2])

    @property
    def length(self) -> float:
        return abs(self.coords[-1] - self.coords[0])

    @property
    def total_expansion(self) -> float:
        return self.end_size / self.start_size

    @property
    def is_simple(self) -> bool:
        """If all sizes are equal, this is a simple grading case"""
        return abs(self.start_size - self.end_size) < TOL

    def get_chop_coords(self) -> FloatListType:
        """Calculates coordinates as will be produced by a Chop"""
        if self.is_simple:
            return np.linspace(self.coords[0], self.coords[-1], num=self.point_count)

        sizes = np.geomspace(self.start_size, self.end_size, num=self.cell_count)
        coords = np.ones(self.point_count) * self.coords[0]

        add = np.cumsum(sizes)

        if self.coords[-1] < self.coords[0]:
            add *= -1

        coords[1:] += add

        return coords

    def get_fitness(self) -> float:
        differences = (self.coords - self.get_chop_coords()) ** 2
        return float(np.sum(differences) / self.cell_count)

    def get_chop(self) -> Chop:
        return Chop(
            length_ratio=self.length,
            count=self.cell_count,
            total_expansion=self.total_expansion,
        )


class Approximator:
    """Takes a list of arbitrary cell sizes and creates Chop objects
    that blockMesh will understand; Tries to minimize differences
    between the rudimentary total expansion + count and actual
    desired (optimized) cell sizes.

    In a possible future scenario where a custom mesher would be employed,
    actual cell sizes could be used without intervention of this object."""

    def __init__(self, coords: FloatListType):
        self.coords = coords
        self.sizes = np.diff(self.coords)
        self.ratios = np.roll(self.sizes, -1)[:-1] / self.sizes[:-1]
        self.count = len(coords) - 1

    @property
    def length(self) -> float:
        return abs(self.coords[-1] - self.coords[0])

    @property
    def is_simple(self) -> bool:
        """Returns True if this is a simple one-chop equal-size scenario"""
        return max(self.sizes) - min(self.sizes) < TOL

    def get_pieces(self, indexes: List[int]) -> List[Piece]:
        """Creates Piece objects between given indexes (adds 0 and -1 automatically);
        does not check if count is smaller than length of indexes"""
        indexes = [0, *indexes]

        pieces = [Piece(self.coords[indexes[i] : indexes[i + 1] + 1]) for i in range(len(indexes) - 1)]
        pieces.append(Piece(self.coords[indexes[-1] :]))

        return pieces

    def get_chops(self, number: int) -> List[Chop]:
        if self.is_simple:
            # don't bother
            return [Chop(count=self.count)]

        # limit number of chops so there's no zero-cell chops
        number = min(number + 1, self.count - 1)

        # brute force: choose the best combination of chops
        # but don't overdo it; just try a couple of scenarios
        refinement = 4
        indexes = np.linspace(0, self.count + 1, num=number * refinement, dtype="int")[1:-1]

        best_fit = 1e12
        best_scenario: Sequence[int] = [0] * (number - 1)

        combinations = itertools.combinations(indexes, r=number - 2)
        for scenario in combinations:
            try:
                pieces = self.get_pieces(scenario)  # type:ignore
                fit = sum(piece.get_fitness() for piece in pieces)
                if fit < best_fit:
                    best_fit = fit
                    best_scenario = scenario
            except (IndexError, ValueError):
                # obviously not the best scenario, eh?
                continue

        pieces = self.get_pieces(best_scenario)  # type:ignore

        chops = [piece.get_chop() for piece in pieces]

        # normalize length ratios
        length = self.length
        for chop in chops:
            chop.length_ratio = chop.length_ratio / length

        return chops
