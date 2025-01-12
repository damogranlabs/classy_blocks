import dataclasses

from classy_blocks.grading.autograding.distributor import DistributorBase


@dataclasses.dataclass
class SmoothDistributor(DistributorBase):
    def get_ideal_ratios(self):
        # In a smooth grader, we want all cells to be as similar to their neighbours as possible
        return super().get_ideal_ratios()
