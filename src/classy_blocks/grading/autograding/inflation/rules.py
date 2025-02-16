from typing import List, Optional, Tuple

import numpy as np
import scipy.optimize

from classy_blocks.grading.autograding.inflation.layers import LayerStack
from classy_blocks.grading.autograding.inflation.params import InflationParams
from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.autograding.smooth.rules import SmoothRules
from classy_blocks.grading.chop import Chop, ChopData
from classy_blocks.types import FloatListType


class OptimizationLayer:
    def __init__(self, length: float, count: int):
        self.length = length
        self.count = count

        self.length_ratio = 1 / 3
        self.total_expansion = 1

    def get_chop(self) -> Chop:
        return Chop(length_ratio=self.length_ratio, count=self.count, total_expansion=self.total_expansion)

    def get_data(self) -> ChopData:
        return self.get_chop().calculate(self.length)


class OptimizationStack:
    def __init__(self, params: InflationParams, length: float, count: int):
        self.params = params
        self.length = length

        # initialize layers as simple uniformly graded chops - this is the
        # safest way to start, optimization will change them to something more appropriate

        # set counts by guessing from bl_thickness and buffer size
        # TODO: correct for count < inflation + buffer + 1
        inflation_count = self.params.inflation_count
        buffer_count = self.params.buffer_count

        self.inflation_layer = OptimizationLayer(self.length, inflation_count)
        self.buffer_layer = OptimizationLayer(self.length, buffer_count)
        self.bulk_layer = OptimizationLayer(self.length, count - inflation_count)

    @property
    def default_values(self):
        return [0.3, 1, 0.5, 1, 1]

    @property
    def bounds(self):
        return [
            [0.1, 0.9],
            [1, 2 * self.params.c2c_expansion],
            [0.1, 0.9],
            [1, 2 * self.params.buffer_expansion],
            [0.1, 99],
        ]

    def update_values(self, values: FloatListType):
        # Values:
        # 0 inflation length ratio
        # 1 inflation total expansion
        # 2 buffer length ratio, relative to remaining length
        # 3 buffer total expansion
        # 4 bulk total expansion
        self.inflation_layer.length_ratio = values[0]
        self.inflation_layer.total_expansion = values[1]

        self.buffer_layer.length_ratio = (1 - values[0]) * values[2]
        self.buffer_layer.total_expansion = values[3]

        self.bulk_layer.total_expansion = values[4]
        self.bulk_layer.length_ratio = 1 - self.inflation_layer.length_ratio - self.buffer_layer.length_ratio

    def get_criteria(self, size_after: float) -> FloatListType:
        # After values have been updated, calculate the new situation
        # and return the new values in a list
        inflation_data = self.inflation_layer.get_data()
        buffer_data = self.buffer_layer.get_data()
        bulk_data = self.bulk_layer.get_data()

        return np.sum(
            np.array(
                [
                    # boundary layer size
                    (self.params.first_cell_size - inflation_data.start_size) ** 2,
                    # continuity
                    ((inflation_data.end_size - buffer_data.start_size) / self.params.buffer_expansion) ** 2,
                    ((buffer_data.end_size - bulk_data.start_size) / self.params.buffer_expansion) ** 2,
                    ((bulk_data.end_size - size_after) / self.params.buffer_expansion) ** 2,
                ]
            )
        )

    def get_chops(self) -> List[Chop]:
        return [self.inflation_layer.get_chop(), self.buffer_layer.get_chop(), self.bulk_layer.get_chop()]


class InflationRules(SmoothRules):
    """See description of InflationGrader"""

    # TODO: refactor to a reasonable number of 'if' clauses

    def __init__(self, params: InflationParams):
        self.params = params

        # shortcuts
        self.first_cell_size = params.first_cell_size
        self.bulk_cell_size = params.bulk_cell_size
        self.c2c_expansion = params.c2c_expansion
        self.bl_thickness_factor = params.bl_thickness_factor
        self.buffer_expansion = params.buffer_expansion

        # use SmoothGrader's logic for bulk chops
        self.cell_size = self.bulk_cell_size

    def get_stack(self, length: float, size_after: Optional[float] = None) -> LayerStack:
        if size_after is None:
            size_after = self.params.bulk_cell_size

        return LayerStack.construct(self.params, length, size_after)

    def get_count(self, length: float, starts_at_wall: bool, ends_at_wall: bool):
        if not (starts_at_wall or ends_at_wall):
            return super().get_count(length, False, False)

        if starts_at_wall and ends_at_wall:
            stack = self.get_stack(length / 2)
            return 2 * stack.count

        stack = self.get_stack(length)
        return stack.count

    def get_sizes(self, info: WireInfo) -> Tuple[float, float]:
        size_before = info.size_before
        if size_before is None:
            if info.starts_at_wall:
                size_before = self.first_cell_size
            else:
                size_before = self.cell_size

        size_after = info.size_after
        if size_after is None:
            if info.ends_at_wall:
                size_after = self.first_cell_size
            else:
                size_after = self.cell_size

        return size_before, size_after

    def is_squeezed(self, count: int, info: WireInfo) -> bool:
        if not (info.starts_at_wall or info.ends_at_wall):
            return super().is_squeezed(count, info)

        length = info.length

        if info.starts_at_wall and info.ends_at_wall:
            length = length / 2

        stack = self.get_stack(length, info.size_after)

        return stack.count < count

        if stack.count < count:
            return True

        if len(stack.layers) < 3:
            return True

        return False

    def get_inflation_chops(self, count: int, length: float) -> List[Chop]:
        """Used when there's not enough room for even the inflation layer"""
        uniform_size = length / count
        if uniform_size > self.first_cell_size:
            # adjust c2c_expansion so that first_cell_size will be maintained
            chop = Chop(start_size=self.first_cell_size, count=count)
        else:
            # just make all cells uniform, already the first is too thin
            # (will avoid making bad aspect ratio cells)
            chop = Chop(count=count)

        return [chop]

    def distribute_cells(self, length: float, count: int, size_after: float) -> List[Chop]:
        # create three chops and optimize them into proper shape
        stack = OptimizationStack(self.params, length, count)

        def fopt(values):
            stack.update_values(values)
            crit = stack.get_criteria(size_after)
            return crit

        # results = scipy.optimize.least_squares(fopt, stack.default_values, bounds=stack.bounds, xtol=0.1)
        results = scipy.optimize.minimize(fopt, stack.default_values, bounds=stack.bounds)
        if not results.success:
            raise RuntimeError(results.message)

        return stack.get_chops()

    def get_chops(self, count, info: WireInfo) -> List[Chop]:
        if not (info.starts_at_wall or info.ends_at_wall):
            # bulk blocks - use SmoothGrader's logic
            return super().get_chops(count, info)

        size_before, size_after = self.get_sizes(info)

        # make every wire start at wall and invert at the end if needed
        if info.ends_at_wall:
            size_before, size_after = size_after, size_before

        length = info.length

        # if length < self.params.bl_thickness + self.params.buffer_thickness:
        if self.is_squeezed(count, info):
            chops = self.get_inflation_chops(count, info.length)
        else:
            chops = self.distribute_cells(length, count, size_after)

        if info.starts_at_wall:
            return chops

        # invert chops
        inverted_chops = []
        for chop in reversed(chops):
            data = chop.calculate(info.length)
            inverted_chops.append(
                Chop(length_ratio=chop.length_ratio, count=chop.count, total_expansion=1 / data.total_expansion)
            )
        return inverted_chops

    def get_squeezed_chops(self, count: int, info: WireInfo) -> List[Chop]:
        return self.get_chops(count, info)
