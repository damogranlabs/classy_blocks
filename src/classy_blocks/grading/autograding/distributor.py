import abc
import dataclasses
from typing import List

import numpy as np
import scipy.optimize

from classy_blocks.grading.autograding.approximator import Approximator
from classy_blocks.grading.chop import Chop
from classy_blocks.types import FloatListType
from classy_blocks.util.constants import TOL


@dataclasses.dataclass
class DistributorBase(abc.ABC):
    """Algorithm that creates chops from given count and sizes;
    first distributes cells using a predefined 'ideal' sizes and ratios,
    then optimizes their *individual* size to get as close to that ideal as possible.

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
        # TODO: repeated code within Approximator! un-repeat
        lengths = np.diff(coords)
        return np.roll(lengths, -1)[:-1] / lengths[:-1]

    def get_ideal_ratios(self) -> FloatListType:
        """Returns desired cell-to-cell ratios"""
        return np.ones(self.count + 1)

    def get_raw_coords(self) -> FloatListType:
        # 'count' denotes number of 'intervals' so add one more point to get points;
        # first and last cells are added artificially ('ghost cells') to calculate
        # proper expansion ratios and sizes
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

            # to prevent 'flipping' over and producing zero or negative length, scale with e^-ratio
            difference = -(self.get_actual_ratios(coords) - self.get_ideal_ratios())
            return np.exp(difference) - 1

        scale = min(self.size_before, self.size_after, self.length / self.count) / 100
        tol = scale / 10
        _ = scipy.optimize.least_squares(ratios, coords[2:-2], ftol=tol, x_scale=scale, gtol=tol, xtol=tol)

        # omit the 'ghost' cells
        print(coords)
        return coords[1:-1]

    @property
    def is_simple(self) -> bool:
        # Don't overdo basic, simple-graded blocks
        base_size = self.length / self.count

        # TODO: use a more relaxed criterion?
        return base_size - self.size_before < TOL and base_size - self.size_after < 10 * TOL

    def get_chops(self, pieces: int) -> List[Chop]:
        approximator = Approximator(self.get_smooth_coords())
        return approximator.get_chops(pieces)

    def get_last_size(self) -> float:
        coords = self.get_smooth_coords()
        return coords[-2] - coords[-3]
