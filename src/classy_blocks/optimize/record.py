import dataclasses
from typing import Literal

MinimizationMethodType = Literal["SLSQP", "L-BFGS-B", "Nelder-Mead", "Powell"]

TerminationReason = Literal["running", "abs", "rel", "limit"]


@dataclasses.dataclass
class OptimizationData:
    # maximum number of iterations; the optimizer will quit at the last iteration
    # regardless of achieved quality
    max_iterations: int = 20
    # absolute tolerance; if overall grid quality stays within given interval
    # between iterations the optimizer will quit
    abs_tol: float = 10
    # relative tolerance; if relative change between iterations is less than
    # given, the optimizer will quit
    rel_tol: float = 0.01
    # method that will be used for scipy.optimize.minimize
    # (only those that support all required features are valid)
    method: MinimizationMethodType = "SLSQP"
    # relaxation base; every subsequent iteration will increase under-relaxation
    # until it's larger than relaxation_threshold; then it will fix it to 1.
    # Relaxation is identical to OpenFOAM's field relaxation
    relaxation: float = 1
    relaxation_threshold: float = 0.9

    # print more-or-less pretty info about optimization
    report: bool = True
    # convergence tolerance for a single joint
    # as passed to scipy.optimize.minimize
    clamp_tol: float = 1e-3


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
        return (self.grid_initial - self.abs_improvement) / self.grid_initial


@dataclasses.dataclass
class ClampRecord:
    vertex_index: int
    # clamp quality is taken from fquality in optimize_clamp();
    # fquality from optimize_clamp() decides whether it's grid or junction quality
    grid_initial: float
    junction_initial: float
    junction_final: float = 0
    grid_final: float = 0
    rolled_back: bool = False
    error_message: str = ""

    @property
    def improvement(self) -> float:
        return self.grid_initial - self.grid_final
