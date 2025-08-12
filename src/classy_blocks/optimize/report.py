import abc

from classy_blocks.optimize.record import ClampRecord, IterationRecord, OptimizationRecord
from classy_blocks.util.tools import report


class OptimizationReporterBase(abc.ABC):
    @abc.abstractmethod
    def iteration_start(self, iteration_no: int, relaxation: float) -> None:
        pass

    @abc.abstractmethod
    def clamp_start(self, crecord: ClampRecord) -> None:
        pass

    @abc.abstractmethod
    def clamp_end(self, crecord: ClampRecord) -> None:
        pass

    @abc.abstractmethod
    def iteration_end(self, srecord: IterationRecord) -> None:
        pass

    @abc.abstractmethod
    def optimization_end(self, orecord: OptimizationRecord) -> None:
        pass


class SilentReporter(OptimizationReporterBase):
    def iteration_start(self, iteration_no: int, relaxation: float) -> None:
        pass

    def clamp_start(self, crecord: ClampRecord) -> None:
        pass

    def clamp_end(self, crecord: ClampRecord) -> None:
        pass

    def iteration_end(self, srecord: IterationRecord) -> None:
        pass

    def optimization_end(self, orecord: OptimizationRecord) -> None:
        pass


class TextReporter(OptimizationReporterBase):
    def iteration_start(self, iteration_no: int, relaxation: float) -> None:
        report(f"Optimization iteration {iteration_no}")
        report(f"Relaxation: {relaxation:.4f}")
        report("{:6s}".format("Vertex"), end="")
        report("{:>12s}".format("Initial"), end="")
        report("{:>12s}".format("Improvement"), end="")
        report("{:>12s}".format("Final"), end="")
        report("{:>12s}".format("Status"))

    def clamp_start(self, crecord: ClampRecord) -> None:
        report(f"{crecord.vertex_index:>6}", end="")
        report(f"{crecord.grid_initial:12.3e}", end="")

    def clamp_end(self, crecord: ClampRecord) -> None:
        report(f"{crecord.improvement:12.0f}", end="")
        report(f"{crecord.grid_final:12.3e}", end="")

        if crecord.rolled_back:
            report(f" Rollback ({crecord.error_message})", end="")

        report("")  # a.k.a. new line

    def iteration_end(self, srecord: IterationRecord) -> None:
        report(f"Iteration {srecord.iteration} finished.", end=" ")
        report(f"Improvement: {srecord.abs_improvement:.0f}", end="")
        report(f" ({srecord.grid_initial:.3e} > {srecord.grid_final:.3e})")

    def optimization_end(self, orecord: OptimizationRecord) -> None:
        message = {
            "abs": "Absolute tolerance reached",
            "rel": "Relative tolerance reached",
            "limit": "Iteration limit hit",
        }[orecord.termination]
        report(f"{message}, stopping optimization.")
        report(
            f"Improvement: {orecord.grid_initial:.3e} > {orecord.grid_final:.3e} ({100 * orecord.rel_improvement:.0f}%)"
        )
        report(f"Duration: {int(orecord.time_end - orecord.time_start)} s")
