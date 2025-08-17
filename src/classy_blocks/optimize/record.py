import dataclasses
from typing import Literal

MinimizationMethodType = Literal["SLSQP", "L-BFGS-B", "Nelder-Mead", "Powell", "trust-constr"]

TerminationReason = Literal["running", "abs", "rel", "limit"]


@dataclasses.dataclass
class OptimizationRecord:
    time_start: float
    grid_initial: float
    termination: TerminationReason = "running"
    grid_final: float = 0
    time_end: float = 0

    @property
    def abs_improvement(self) -> float:
        return self.grid_initial - self.grid_final

    @property
    def rel_improvement(self) -> float:
        return self.abs_improvement / self.grid_initial


@dataclasses.dataclass
class IterationRecord:
    iteration: int
    grid_initial: float
    relaxation: float
    grid_final: float = 0

    @property
    def abs_improvement(self) -> float:
        return self.grid_initial - self.grid_final

    @property
    def rel_improvement(self) -> float:
        return (self.grid_initial - self.grid_final) / self.grid_initial


@dataclasses.dataclass
class ClampRecord:
    vertex_index: int
    # clamp quality is taken from fquality in optimize_clamp();
    grid_initial: float
    grid_final: float = 0
    rolled_back: bool = False
    error_message: str = ""

    @property
    def improvement(self) -> float:
        return self.grid_initial - self.grid_final
