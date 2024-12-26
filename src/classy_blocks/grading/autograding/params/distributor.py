import abc
import dataclasses
from typing import List

import numpy as np
import scipy.interpolate
import scipy.optimize

from classy_blocks.grading.autograding.params.base import sum_length
from classy_blocks.grading.autograding.params.layers import InflationLayer
from classy_blocks.grading.chop import Chop
from classy_blocks.types import FloatListType
from classy_blocks.util.constants import TOL


@dataclasses.dataclass
class DistributorBase(abc.ABC):
    """Algorithm that creates chops from given count and sizes;
    first distributes cells using a predefined 'ideal' sizes and ratios,
    then optimizes their individual size to get as close to that ideal as possible.

    Then, when cells are placed, Chops are created so that they produce as similar
    cells to the calculated as possible. Since blockMesh only supports equal
    cell-to-cell expansion for each chop, currently creating a limited number of Chops
    is the only way to go. In a possible future (a.k.a. own meshing script),
    calculated cell sizes could be used directly for meshing."""

    count: int
    size_before: float
    length: float
    size_after: float

    @staticmethod
    def get_actual_ratios(coords: FloatListType) -> FloatListType:
        lengths = np.diff(coords)
        return (lengths[:-1] / np.roll(lengths, -1)[:-1]) ** -1

    @abc.abstractmethod
    def get_ideal_ratios(self) -> FloatListType:
        """Returns desired cell-to-cell ratios"""
        return np.ones(self.count + 1)

    @abc.abstractmethod
    def get_ratio_weights(self) -> FloatListType:
        """Returns weights of cell ratios"""

    def get_raw_coords(self) -> FloatListType:
        # 'count' denotes number of 'intervals' so there must be another point
        return np.concatenate(
            ([-self.size_before], np.linspace(0, self.length, num=self.count + 1), [self.length + self.size_after])
        )

    def get_smooth_coords(self) -> FloatListType:
        coords = self.get_raw_coords()

        # prepare a function for least squares minimization:
        # f(coords) -> actual_ratios / ideal_ratios
        # first two and last two points are fixed

        def ratios(inner_coords):
            coords[2:-2] = inner_coords

            difference = self.get_actual_ratios(coords) - self.get_ideal_ratios()
            return difference * self.get_ratio_weights()

        scale = min(self.size_before, self.size_after, self.length / self.count) / 10
        _ = scipy.optimize.least_squares(ratios, coords[2:-2], method="lm", ftol=scale / 100, x_scale=scale)

        return coords

    @property
    def is_simple(self) -> bool:
        # Don't overdo basic, simple-graded blocks
        base_size = self.length / self.count

        # TODO: use a more relaxed criterion?
        return base_size - self.size_before < TOL and base_size - self.size_after < TOL

    def get_chops(self, pieces: int) -> List[Chop]:
        if self.is_simple:
            return [Chop(count=self.count)]

        coords = self.get_smooth_coords()
        sizes = np.diff(coords)
        ratios = self.get_actual_ratios(coords)

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
            chop_count = indexes[i + 1] - indexes[i]
            avg_ratio = ratio ** (1 / chop_count)
            length = sum_length(sizes[index], chop_count, avg_ratio)
            chop = Chop(length_ratio=length, total_expansion=ratio, count=chop_count)
            chops.append(chop)

        # normalize length ratios
        sum_ratio = sum([chop.length_ratio for chop in chops])
        for chop in chops:
            chop.length_ratio = chop.length_ratio / sum_ratio

        return chops


@dataclasses.dataclass
class SmoothDistributor(DistributorBase):
    def get_ideal_ratios(self):
        # In a smooth grader, we want all cells to be as similar to their neighbours as possible
        return super().get_ideal_ratios()

    def get_ratio_weights(self):
        # Enforce stricter policy on size_before and size_after
        weights = np.ones(self.count + 1)
        for i in (0, 1, 2, 3):
            w = 2 ** (3 - i)
            weights[i] = w
            weights[-i - 1] = w

        return weights


@dataclasses.dataclass
class InflationDistributor(SmoothDistributor):
    c2c_expansion: float
    bl_thickness_factor: int

    @property
    def is_simple(self) -> bool:
        return False

    def get_ideal_ratios(self):
        # Ideal growth ratio in boundary layer is user-specified c2c_expansion;
        inflation_layer = InflationLayer(self.size_before, self.c2c_expansion, self.bl_thickness_factor, 1e12)
        inflation_count = inflation_layer.count
        print(f"Inflation count: {inflation_count}")

        ratios = super().get_ideal_ratios()

        ratios[:inflation_count] = self.c2c_expansion
        print(ratios)

        return ratios

    def get_ratio_weights(self):
        return super().get_ratio_weights()

    def _get_ratio_weights(self):
        return np.ones(self.count + 1)
