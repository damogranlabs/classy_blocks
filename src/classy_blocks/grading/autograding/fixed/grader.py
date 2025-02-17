from classy_blocks.grading.autograding.fixed.rules import FixedCountRules
from classy_blocks.grading.autograding.grader import GraderBase
from classy_blocks.mesh import Mesh


class FixedCountGrader(GraderBase):
    """The simplest possible mesh grading: use a constant cell count for all axes on all blocks;
    useful during mesh building and some tutorial cases"""

    def __init__(self, mesh: Mesh, count: int = 8):
        super().__init__(mesh, FixedCountRules(count))
