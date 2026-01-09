from classy_blocks.assemble.dump import AssembledDump
from classy_blocks.cbtyping import DirectionType
from classy_blocks.grading.analyze.row import Row
from classy_blocks.grading.define.chop import Chop
from classy_blocks.grading.graders.auto import AutoGraderMixin
from classy_blocks.grading.graders.manager import AxisGrader, GradingManager
from classy_blocks.items.block import Block
from classy_blocks.mesh import Mesh


class FixedAxisGrader(AxisGrader):
    def __init__(self, block: Block, direction: DirectionType, count: int):
        self.count = count
        super().__init__(block, direction)

    def get_chops(self):
        return [Chop(count=self.count)]


class FixedCountGrader(GradingManager, AutoGraderMixin):
    def __init__(self, mesh: Mesh, count: int = 5):
        self.count = count
        mesh.assemble()

        assert isinstance(mesh.dump, AssembledDump)

        super().__init__(mesh.dump, mesh.settings)

    def _grade_row(self, row: Row):
        super()._grade_row(row)

        if row.count == 0:
            entry = row.entries[0]
            axis_grader = FixedAxisGrader(entry.block, entry.heading, self.count)
            axis_grader.grade()
