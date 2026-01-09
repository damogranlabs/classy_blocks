from classy_blocks.assemble.dump import AssembledDump
from classy_blocks.cbtyping import ChopTakeType
from classy_blocks.grading.analyze.row import Row
from classy_blocks.grading.graders.auto import AutoGraderMixin
from classy_blocks.grading.graders.fixed import FixedAxisGrader
from classy_blocks.grading.graders.manager import GradingManager
from classy_blocks.mesh import Mesh


class SimpleGrader(GradingManager, AutoGraderMixin):
    def __init__(self, mesh: Mesh, cell_size: float, take: ChopTakeType = "avg"):
        self.cell_size = cell_size
        self.take = take
        mesh.assemble()

        assert isinstance(mesh.dump, AssembledDump)

        super().__init__(mesh.dump, mesh.settings)

    def _get_row_count(self, row: Row):
        return max(1, int(self._get_row_length(row) / self.cell_size))

    def _grade_row(self, row: Row):
        super()._grade_row(row)

        if row.count == 0:
            entry = row.entries[0]
            axis_grader = FixedAxisGrader(entry.block, entry.heading, self._get_row_count(row))
            axis_grader.grade()
