import dataclasses
from typing import List

from classy_blocks.util.constants import VBIG, VSMALL
from classy_blocks.util.tools import report


@dataclasses.dataclass
class ClampOptimizationData:
    """Quality tracking pre/post iteration"""

    index: int
    grid_initial: float
    junction_initial: float
    junction_final: float = VBIG
    grid_final: float = VBIG

    skipped: bool = False
    rolled_back: bool = False

    def report_start(self) -> None:
        report(f"{self.index:>6}", end="   ")
        report(f"{self.grid_initial:.3e}", end="   ")
        report(f"{self.junction_initial:.3e}", end="   ")

    def report_end(self) -> None:
        report(f"{self.improvement: >11.0f}", end="   ")
        report(f"{self.grid_final:.3e}", end="   ")

        comment = ""
        if self.skipped:
            comment = "Skip"
        elif self.rolled_back:
            comment = "Rollback"

        report(comment)

    def undo(self) -> None:
        self.junction_final = self.junction_initial
        self.grid_final = self.grid_initial

    def rollback(self) -> None:
        self.rolled_back = True
        self.undo()

    def skip(self) -> None:
        self.skipped = True
        self.undo()

    @property
    def improvement(self) -> float:
        return self.grid_initial - self.grid_final


class IterationData:
    """Data about a single iteration's progress"""

    def __init__(self, index: int, initial_quality: float):
        self.index = index
        self.initial_quality = initial_quality
        self.final_quality: float = VBIG

    @property
    def improvement(self) -> float:
        if abs(self.initial_quality - self.final_quality) < VSMALL:
            return VSMALL

        return self.initial_quality - self.final_quality

    def report_begin(self):
        report(f"Optimization iteration {self.index + 1}:")
        # headers
        report("Vertex     Initial       Local   Improvement       Final   Status")

    def report_end(self):
        report(f"Iteration {self.index + 1} finished.", end=" ")
        report(f"Improvement: {self.initial_quality - self.final_quality:.0f}", end="")
        report(f" ({self.initial_quality:.3e} > {self.final_quality:.3e})")


class IterationDriver:
    """Bookkeeping: iterations, results, quality and whatnot"""

    def __init__(self, max_iterations: int, tolerance: float):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self.iterations: List[IterationData] = []

    def begin_iteration(self, quality: float) -> IterationData:
        iteration = IterationData(len(self.iterations), quality)
        iteration.report_begin()

        self.iterations.append(iteration)

        return iteration

    def end_iteration(self, quality: float) -> None:
        iteration = self.iterations[-1]

        iteration.final_quality = quality
        iteration.report_end()

    @property
    def initial_improvement(self) -> float:
        if len(self.iterations) < 1:
            return VBIG

        return self.iterations[0].improvement

    @property
    def last_improvement(self) -> float:
        if len(self.iterations) < 2:
            return self.initial_improvement

        return self.iterations[-1].improvement

    @property
    def converged(self) -> bool:
        if len(self.iterations) >= self.max_iterations:
            report("Iteration limit hit, stopping optimization.")
            return True

        if len(self.iterations) < 2:
            # Can't decide without data
            return False

        if self.last_improvement / self.iterations[0].initial_quality < self.tolerance:
            print("Tolerance reached, stopping optimization.")
            return True

        return False
