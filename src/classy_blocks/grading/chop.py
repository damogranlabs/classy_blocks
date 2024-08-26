import dataclasses
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from classy_blocks.grading import relations as rel
from classy_blocks.types import ChopPreserveType


@dataclasses.dataclass
class ChopRelation:
    """A container that links a pair of inputs to a
    grading calculator function that outputs a new value"""

    output: str

    input_1: str
    input_2: str

    function: Callable[[float, float, float], Union[int, float]]

    @property
    def inputs(self) -> Set[str]:
        """Input parameters for this chop function"""
        return {self.input_1, self.input_2}

    @classmethod
    def from_function(cls, function: Callable):
        """Create this object from given callable (relations functions)"""
        try:
            # function name is assembled as `get_<result>__<param1>__<param2>`
            data = function.__name__.split("__")
            result_param_name = data[0][4:]
            param1 = data[1]
            param2 = data[2]

            return cls(result_param_name, param1, param2, function)
        except Exception as err:
            raise RuntimeError(f"Invalid function name or unexpected parameter names: {function.__name__}") from err

    @staticmethod
    @lru_cache(maxsize=1)
    def get_possible_combinations() -> List["ChopRelation"]:
        calculation_functions = rel.get_calculation_functions()

        return [ChopRelation.from_function(f) for _, f in calculation_functions.items()]


@dataclasses.dataclass
class Chop:
    """A single 'chop' represents a division in Grading object;
    user-provided arguments are stored in this object and used
    for creation of Gradient object"""

    length_ratio: float = 1.0
    start_size: Optional[float] = None
    c2c_expansion: Optional[float] = None
    count: Optional[int] = None
    end_size: Optional[float] = None
    total_expansion: Optional[float] = None
    invert: bool = False
    preserve: ChopPreserveType = "c2c_expansion"

    def __post_init__(self) -> None:
        # default: take c2c_expansion=1 if there's less than 2 parameters given
        grading_params = [self.start_size, self.end_size, self.count, self.total_expansion, self.c2c_expansion]
        if len(grading_params) - grading_params.count(None) < 2:
            if self.c2c_expansion is None:
                self.c2c_expansion = 1

        # also: count can only be an integer
        if self.count is not None:
            self.count = max(int(self.count), 1)

        # stored results from self.calculate() method;
        # this will be used for creating new chops taking self.preserve
        # into account. User input (chop arguments) should not be overridden because
        # if mesh is modified (optimization etc.) all numbers will be wrong.
        self.results: Dict[Union[ChopPreserveType, str], Union[float, int]] = dict()

    def calculate(self, length: float) -> Tuple[int, float]:
        """Calculates cell count and total expansion ratio for this chop
        by calling functions that take known variables and return new values"""
        data = dataclasses.asdict(self)
        self.results = data
        calculated: Set[str] = set()

        for key in self.results.keys():
            if data[key] is not None:
                calculated.add(key)

        for _ in range(12):
            if {"count", "total_expansion", "c2c_expansion", "start_size", "end_size"}.issubset(calculated):
                self.results["count"] = int(self.results["count"])

                if self.invert:
                    data["total_expansion"] = 1 / data["total_expansion"]

                return data["count"], data["total_expansion"]

            for chop_rel in ChopRelation.get_possible_combinations():
                output = chop_rel.output
                inputs = chop_rel.inputs
                function = chop_rel.function

                if output in calculated:
                    # this value is already calculated, go on
                    continue

                if inputs.issubset(calculated):
                    # value is not yet calculated but parameters are available
                    data[output] = function(length, data[chop_rel.input_1], data[chop_rel.input_2])
                    calculated.add(output)

        raise ValueError(f"Could not calculate count and grading for given parameters: {data}")

    def copy_preserving(self, inverted: bool = False) -> "Chop":
        """Creates a copy of this Chop with equal count but
        sets other parameters from current data so that
        the correct start/end size or c2c is maintained"""
        args = dataclasses.asdict(self)
        args["count"] = self.results["count"]

        for arg in ["total_expansion", "c2c_expansion", "start_size", "end_size"]:
            args[arg] = None

        # TODO: get rid of if-ing
        if self.preserve == "start_size":
            if inverted:
                args["end_size"] = self.results["start_size"]
            else:
                args["start_size"] = self.results["start_size"]
        elif self.preserve == "end_size":
            if inverted:
                args["start_size"] = self.results["end_size"]
            else:
                args["end_size"] = self.results["end_size"]
        else:
            if inverted:
                args["c2c_expansion"] = 1 / self.results["c2c_expansion"]
            else:
                args["c2c_expansion"] = self.results["c2c_expansion"]

        return Chop(**args)
