import dataclasses
import inspect

from typing import Optional, Callable, Union, Tuple, Set

from classy_blocks.types import ChopTakeType
from classy_blocks.grading import relations as rel


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
    def from_function(cls, name: str, function: Callable):
        """Create this object from given name and callable as returned by
        inspect.getmembers() thingy"""
        # function name is assembled as
        # get_<result>__<param1>__<param2>
        data = name.split(sep="__")

        return cls(data[0][4:], data[1], data[2], function)


# gather available functions for calculation of grading parameters
functions = [ChopRelation.from_function(*member) for member in inspect.getmembers(rel, inspect.isfunction)]


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
    take: ChopTakeType = "avg"

    def __post_init__(self):
        # default: take c2c_expansion=1 if there's less than 2 parameters given
        grading_params = [self.start_size, self.end_size, self.count, self.total_expansion, self.c2c_expansion]
        if len(grading_params) - grading_params.count(None) < 2:
            if self.c2c_expansion is None:
                self.c2c_expansion = 1

        # also: count can only be an integer
        if self.count is not None:
            self.count = max(int(self.count), 1)

    def calculate(self, length: float) -> Tuple[int, float]:
        """Calculates cell count and total expansion ratio for this chop
        by calling functions that take known variables and return new values"""
        data = dataclasses.asdict(self)
        calculated: Set[str] = set()

        for key in data.keys():
            if data[key] is not None:
                calculated.add(key)

        for _ in range(20):
            if {"count", "total_expansion"}.issubset(calculated):
                count = int(data["count"])
                total_expansion = data["total_expansion"]

                if self.invert:
                    return count, 1 / total_expansion

                return count, total_expansion

            for crel in functions:
                output = crel.output
                inputs = crel.inputs
                function = crel.function

                if output in calculated:
                    # this value is already calculated, go on
                    continue

                if inputs.issubset(calculated):
                    # value is not yet calculated but parameters are available
                    data[output] = function(length, data[crel.input_1], data[crel.input_2])
                    calculated.add(output)

        raise ValueError(f"Could not calculate count and grading for given parameters: {data}")
