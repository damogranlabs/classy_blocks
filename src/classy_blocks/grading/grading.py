""" In theory, combination of three of these 6 values can be specified:
 - Total length
 - Number of cells
 - Total expansion ratio
 - Cell-to-cell expansion ratio
 - Width of start cell
 - Width of end cell

Also in reality, some are very important:
 - Width of start cell: required to keep y+ <= 1
 - Cell-to-cell expansion ratio: 1 < recommended <= 1.2
 - Number of cells: the obvious meshing parameter
 - Width of end cell: matching with cell sizes of the next block

Possible Combinations to consider are:
 - start_size & c2c_expansion
 - start_size & count
 - start_size & end_size
 - end_size & c2c_expansion
 - end_size & count
 - count & c2c_expansion

Some nomenclature to avoid confusion
Grading: the whole grading specification (length, count, expansion ratio)
length: length of an edge we're dealing with
count: number of cells in a block for a given direction
total_expansion: ratio between last/first cell size
c2c_expansion: ratio between sizes of two neighbouring cells
start_size: size of the first cell in a block
end_size: size of the last cell in a block
division: an entry in simpleGrading specification in blockMeshDict

calculations meticulously transcribed from the blockmesh grading calculator:
https://gitlab.com/herpes-free-engineer-hpe/blockmeshgradingweb/-/blob/master/calcBlockMeshGrading.coffee
(since block length is always known, there's less wrestling but the calculation principle is similar) """
import copy
import inspect
import warnings
import math

from typing import Tuple, List

from classy_blocks.grading.chop import Chop
from classy_blocks.grading import calculator as gc
from classy_blocks.util import constants

# gather available functions for calculation of grading parameters
functions = []  # list [ [return_value, {parameters}, function], ... ]

for m in inspect.getmembers(gc):
    if inspect.isfunction(m[1]):
        # function name is assembled as
        # get_<result>__<param1>__<param2>
        data = m[0].split(sep="__")
        functions.append([data[0][4:], [data[1], data[2]], m[1]])


def calculate(length: float, parameters: dict) -> Tuple[int, float]:
    """Calculates cell count and total expansion ratio for a block
    by calling functions that take known variables and return new values"""
    # FIXME: move all this yada yada into Division class or something
    keys = parameters.keys()
    calculated = set()

    for k in keys:
        if parameters[k] is not None:
            calculated.add(k)

    for _ in range(20):
        for fdata in functions:
            freturn = fdata[0]
            fparams = fdata[1]
            ffunction = fdata[2]
            if freturn in calculated:
                # this value is already calculated, go on
                continue

            if set(fparams).issubset(calculated):
                parameters[freturn] = ffunction(length, parameters[fparams[0]], parameters[fparams[1]])
                calculated.add(freturn)

        if {"count", "total_expansion"}.issubset(calculated):
            return int(parameters["count"]), parameters["total_expansion"]

    raise ValueError(f"Could not calculate count and grading for given parameters: {parameters}")


class Grading:
    """Grading specification for a single edge"""
    def __init__(self, length:float):
        # "multi-grading" specification according to:
        # https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading
        self.specification:List[List] = []  # a list of lists [length ratio, count ratio, total expansion]

        self.length = length

    def add_chop(self, chop:Chop) -> None:
        """Add a grading division to block specification.
        Use length_ratio for multigrading (see documentation).
        Available grading parameters are:
         - start_size: width of start cell
         - end_size: width of end cell
         - count: cell count in given direction
         - c2c_expansion: cell-to-cell expansion ratio (default=1)
         - total_expansion: ratio between first and last cell size

        You must specify start_size and/or count.
        c2c_expansion is optional - will be used to create graded cells
        and will default to 1 if not provided.

        To reverse grading, use invert=True.

        Documentation:
        https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading"""
        assert 0 < chop.length_ratio <= 1

        # default: take c2c_expansion=1 if there's less than 2 parameters given
        grading_params = [chop.start_size, chop.end_size, chop.c2c_expansion, chop.count]
        if grading_params.count(None) > len(grading_params) - 2:
            chop.c2c_expansion = 1

        # also: count can only be an integer
        if chop.count is not None:
            chop.count = int(chop.count)

        length = self.length * chop.length_ratio

        # blockMesh needs two numbers regardless of user's input:
        # number of cells and total expansion ratio.
        # those two can be calculated from count and c2c_expansion
        # so calculate those first
        count, total_expansion = calculate(
            length,
            {
                "start_size": chop.start_size,
                "end_size": chop.end_size,
                "c2c_expansion": chop.c2c_expansion,
                "count": chop.count,
                "total_expansion": chop.total_expansion,
            },
        )

        if chop.invert:
            total_expansion = 1 / total_expansion

        self.specification.append([chop.length_ratio, count, total_expansion])

    @property
    def inverted(self) -> 'Grading':
        """Returns this grading but inverted
        in case neighbours are defined upside-down"""
        if len(self.specification) == 0:
            return  self # nothing to invert

        g_inv = copy.deepcopy(self)

        # divisions:
        # a list of lists [length ratio, count ratio, total expansion]

        # reverse the list first
        g_inv.specification.reverse()

        # then do 1/total_expansion
        for i, div in enumerate(g_inv.specification):
            g_inv.specification[i][2] = 1/div[2]

        return g_inv

    @property
    def count(self) -> int:
        """Return number of cells, summed over all sub-divisions"""
        return sum(d[1] for d in self.specification)

    @property
    def is_defined(self) -> bool:
        """Return True is grading is defined;
        It is if there's at least one division added"""
        return len(self.specification) > 0

    @property
    def description(self) -> str:
        """Output string for blockMeshDict"""
        if not self.is_defined:
            raise ValueError(f"Grading not defined: {self}")

        if len(self.specification) == 1:
            # its a one-number simpleGrading:
            return str(self.specification[0][2])

        # multi-grading: make a nice list
        # FIXME: make a nicer list
        length_ratio_sum = 0
        out = "("

        for spec in self.specification:
            out += f"({spec[0]} {spec[1]} {spec[2]})"
            length_ratio_sum += spec[0]

        out += ")"

        if not math.isclose(length_ratio_sum, 1, rel_tol=constants.TOL):
            warnings.warn(f"Length ratio doesn't add up to 1: {length_ratio_sum}")

        return out

    def __eq__(self, other):
        # this works theoretically but numerics will probably ruin the party:
        # return self.specification == other.specification

        # so just compare number-by-number
        for i, this_spec in enumerate(self.specification):
            other_spec = other.specification[i]

            for j, this_value in enumerate(this_spec):
                other_value = other_spec[j]

                if not math.isclose(this_value, other_value, rel_tol=constants.TOL):
                    return False

        return True
