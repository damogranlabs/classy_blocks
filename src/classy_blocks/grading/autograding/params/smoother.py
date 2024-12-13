import dataclasses
from typing import List

import numpy as np
import scipy.interpolate
import scipy.optimize

from classy_blocks.grading.chop import Chop
from classy_blocks.types import FloatListType


@dataclasses.dataclass
class Smoother:
    """A 1-dimensional smoother for calculation of autograding Chops"""

    count: int
    size_before: float
    length: float
    size_after: float

    @staticmethod
    def get_ratios(coords: FloatListType) -> FloatListType:
        lengths = np.diff(coords)
        return (lengths[:-1] / np.roll(lengths, -1)[:-1]) ** -1

    def get_raw_coords(self) -> FloatListType:
        # 'count' denotes number of 'intervals' so there must be another point
        return np.concatenate(
            ([-self.size_before], np.linspace(0, self.length, num=self.count + 1), [self.length + self.size_after])
        )

    def get_smooth_coords(self) -> FloatListType:
        coords = self.get_raw_coords()

        # prepare a function for least squares minimization:
        # f(coords) -> cell ratios (next/prev)
        # first two and last two points are fixed
        # the goal is to have

        def ratios(inner_coords):
            coords[2:-2] = inner_coords

            return self.get_ratios(coords)

        scale = min(self.size_before, self.size_after, self.length / self.count) / 10
        _ = scipy.optimize.least_squares(ratios, coords[2:-2], method="lm", ftol=scale / 100, x_scale=scale)

        return coords

    def get_chops(self, pieces: int) -> List[Chop]:
        coords = self.get_smooth_coords()
        ratios = self.get_ratios(coords)

        count = len(ratios) - 1

        # create a piecewise linear function from a number of chosen indexes and their respective c2c ratios
        # then optimize indexes to obtain best fit
        # fratios = scipy.interpolate.interp1d(range(len(ratios)), ratios)

        # def get_piecewise(indexes: List[int]) -> Callable:
        #     values = np.take(ratios, indexes)
        #     return scipy.interpolate.interp1d(indexes, values)

        # def get_fitness(indexes: List[int]) -> float:
        #     fitted = get_piecewise(indexes)(range(len(ratios)))

        #     ss_tot = np.sum((ratios - np.mean(ratios)) ** 2)
        #     ss_res = np.sum((ratios - fitted) ** 2)

        #     return 1 - (ss_res / ss_tot)

        # print(get_fitness([0, 9, 10, count]))
        # print(get_fitness(np.linspace(0, count, num=pieces + 1, dtype=int)))

        chops: List[Chop] = []
        indexes = np.linspace(0, count, num=pieces + 1, dtype=int)

        for i, index in enumerate(indexes[:-1]):
            chop_ratios = ratios[index : indexes[i + 1]]
            ratio = np.prod(chop_ratios)
            chop = Chop(length_ratio=1 / pieces, total_expansion=ratio, count=indexes[i + 1] - indexes[i])
            chops.append(chop)

        return chops
