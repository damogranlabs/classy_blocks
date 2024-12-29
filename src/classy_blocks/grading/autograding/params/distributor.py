import abc
import dataclasses
from typing import List

import numpy as np
import scipy.interpolate
import scipy.optimize

from classy_blocks.grading.autograding.params.approximator import Approximator
from classy_blocks.grading.autograding.params.layers import InflationLayer
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

    @abc.abstractmethod
    def get_ratio_weights(self) -> FloatListType:
        """Returns weights of cell ratios"""

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
            difference = -(self.get_ratio_weights() * (self.get_actual_ratios(coords) - self.get_ideal_ratios()))
            return np.exp(difference) - 1

        scale = min(self.size_before, self.size_after, self.length / self.count) / 100
        _ = scipy.optimize.least_squares(ratios, coords[2:-2], ftol=scale / 10, x_scale=scale)

        # omit the 'ghost' cells
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


@dataclasses.dataclass
class SmoothDistributor(DistributorBase):
    def get_ideal_ratios(self):
        # In a smooth grader, we want all cells to be as similar to their neighbours as possible
        return super().get_ideal_ratios()

    def get_ratio_weights(self):
        weights = np.ones(self.count + 1)
        # Enforce stricter policy on the first few cells
        # to match size_before and size_after
        for i in (0, 1, 2, 3):
            w = 2 ** (3 - i)
            weights[i] = w
            weights[-i - 1] = w

        return weights


@dataclasses.dataclass
class InflationDistributor(SmoothDistributor):
    c2c_expansion: float
    bl_thickness_factor: int
    buffer_expansion: float
    bulk_size: float

    @property
    def is_simple(self) -> bool:
        return False

    def get_ideal_ratios(self):
        # TODO: combine this logic and LayerStack;
        # possibly package all parameters into a separate dataclass
        ratios = super().get_ideal_ratios()

        # Ideal growth ratio in boundary layer is user-specified c2c_expansion;
        inflation_layer = InflationLayer(self.size_before, self.c2c_expansion, self.bl_thickness_factor, 1e12)
        inflation_count = inflation_layer.count

        ratios[:inflation_count] = self.c2c_expansion

        # add a buffer layer if needed
        last_inflation_size = inflation_layer.end_size
        if self.bulk_size > self.buffer_expansion * last_inflation_size:
            buffer_count = int(np.log(self.bulk_size / last_inflation_size) / np.log(self.buffer_expansion)) + 1
            ratios[inflation_count : inflation_count + buffer_count] = self.buffer_expansion

        return ratios

    def get_ratio_weights(self):
        # using the same weights as in SmoothDistributor
        # can trigger overflow warnings but doesn't produce
        # better chops; thus, keep it simple
        return np.ones(self.count + 1)
