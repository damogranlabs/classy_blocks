from classy_blocks.grading.autograding.grader import GraderBase
from classy_blocks.grading.autograding.smooth.rules import SmoothRules
from classy_blocks.mesh import Mesh


class SmoothGrader(GraderBase):
    """Parameters for mesh grading for high-Re cases.
    Two chops are added to all blocks; c2c_expansion and and length_ratio
    are utilized to keep cell sizes between blocks consistent
    (as much as possible)"""

    def __init__(self, mesh: Mesh, cell_size: float):
        super().__init__(mesh, SmoothRules(cell_size))
