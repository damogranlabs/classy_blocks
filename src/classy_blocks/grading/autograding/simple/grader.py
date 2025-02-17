from classy_blocks.grading.autograding.grader import GraderBase
from classy_blocks.grading.autograding.simple.rules import SimpleRules
from classy_blocks.mesh import Mesh


class SimpleGrader(GraderBase):
    """Simple mesh grading for high-Re cases.
    A single chop is used that sets cell count based on size.
    Cell sizes between blocks differ as blocks' sizes change."""

    def __init__(self, mesh: Mesh, cell_size: float):
        super().__init__(mesh, SimpleRules(cell_size))
