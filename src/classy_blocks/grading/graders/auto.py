import abc

import numpy as np

from classy_blocks.cbtyping import ChopTakeType
from classy_blocks.grading.analyze.row import Row


class AutoGraderMixin(abc.ABC):
    take: ChopTakeType

    def _get_row_length(self, row: Row) -> float:
        lengths = []

        for entry in row.entries:
            lengths += entry.lengths

        if self.take == "max":
            return max(lengths)

        if self.take == "min":
            return min(lengths)

        return float(np.average(np.array(lengths)))
