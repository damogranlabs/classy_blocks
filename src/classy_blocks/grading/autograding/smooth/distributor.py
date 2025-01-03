import dataclasses

import numpy as np

from classy_blocks.grading.autograding.distributor import DistributorBase


@dataclasses.dataclass
class SmoothDistributor(DistributorBase):
    def get_ideal_ratios(self):
        # In a smooth grader, we want all cells to be as similar to their neighbours as possible
        return super().get_ideal_ratios()

    def get_ratio_weights(self):
        weights = np.ones(self.count + 1)
        # Enforce stricter policy on the first few cells
        # to match size_before and size_after
        for i in range(min(self.count // 2, 4)):
            w = 2 ** (3 - i)
            weights[i] = w
            weights[-i - 1] = w

        return weights
