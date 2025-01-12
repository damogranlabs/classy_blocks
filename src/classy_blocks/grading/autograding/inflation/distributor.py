import dataclasses

import numpy as np

from classy_blocks.grading.autograding.inflation.layers import InflationLayer
from classy_blocks.grading.autograding.smooth.distributor import SmoothDistributor
from classy_blocks.types import FloatListType


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

        ratios[1:inflation_count] = self.c2c_expansion

        # add a buffer layer if needed
        last_inflation_size = inflation_layer.end_size
        if self.bulk_size > self.buffer_expansion * last_inflation_size:
            buffer_count = int(np.log(self.bulk_size / last_inflation_size) / np.log(self.buffer_expansion)) + 1
            ratios[inflation_count : inflation_count + buffer_count] = self.buffer_expansion

        return ratios


class DoubleInflationDistributor(InflationDistributor):
    @staticmethod
    def flip_ratios(ratios: FloatListType) -> FloatListType:
        flipped = 1 / np.flip(ratios)
        count = len(ratios)

        result = np.ones(count)
        result[: count // 2] = ratios[: count // 2]
        result[count // 2 :] = flipped[count // 2 :]

        return result

    def get_ideal_ratios(self):
        # same as super()'s but with an inflation layer at both ends
        single_ratios = super().get_ideal_ratios()

        return self.flip_ratios(single_ratios)
