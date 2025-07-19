"""In theory, combination of three of these 6 values can be specified:
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
(since block length is always known, there's less wrestling but the calculation principle is similar)"""

import dataclasses
import math
import warnings

from classy_blocks.base.exceptions import UndefinedGradingsError
from classy_blocks.cbtyping import GradingSpecType
from classy_blocks.grading.chop import Chop, ChopData
from classy_blocks.util import constants


class Grading:
    """Grading specification for a single edge"""

    def __init__(self, length: float):
        # "multi-grading" specification according to:
        # https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading
        self.length = length  # to be updated when adding/modifying block edges

        # will change specification to match a wire from end to start
        self.inverted: bool = False

        # a list of user-input data
        self.chops: list[Chop] = []
        # results from chop.calculate():
        self._chop_data: list[ChopData] = []

    def add_chop(self, chop: Chop) -> None:
        if not (0 < chop.length_ratio <= 1):
            raise ValueError(f"Length ratio must be between 0 and (including) 1, got {chop.length_ratio}")

        self.chops.append(chop)

    def clear(self) -> None:
        self.chops = []
        self._chop_data = []

    @property
    def chop_data(self) -> list[ChopData]:
        if len(self._chop_data) < len(self.chops):
            # Chops haven't been calculated yet
            self._chop_data = []

            for chop in self.chops:
                self._chop_data.append(chop.calculate(self.length))

        return self._chop_data

    def get_specification(self, inverted: bool) -> list[GradingSpecType]:
        if inverted:
            chops = list(reversed(self.chop_data))
        else:
            chops = self.chop_data

        return [data.get_specification(inverted) for data in chops]

    @property
    def specification(self) -> list[GradingSpecType]:
        # a list of [length_ratio, count, total_expansion] for each chop
        return self.get_specification(self.inverted)

    @property
    def start_size(self) -> float:
        if len(self.chops) == 0:
            raise RuntimeError("start_size requested but no chops defined")

        chop = self.chops[0]
        return chop.calculate(self.length).start_size

    @property
    def end_size(self) -> float:
        if len(self.chops) == 0:
            raise RuntimeError("end_size requested but no chops defined")

        chop = self.chops[-1]
        return chop.calculate(self.length).end_size

    def copy(self, length: float, invert: bool = False) -> "Grading":
        """Creates a new grading with the same chops (counts) on a different length,
        keeping chop.preserve quantity constant;

        the 'length' parameter is the new wire's length;
        'invert' does not set the grading.inverted flag but flips the original value"""
        new_grading = Grading(length)

        for data in self.chop_data:
            # take count from calculated chops;
            # it is of utmost importance it stays the same
            old_data = dataclasses.asdict(data)
            new_args = {
                "length_ratio": data.length_ratio,
                "count": old_data["count"],
                data.preserve: old_data[data.preserve],
                "preserve": data.preserve,
                "take": data.take,
            }

            new_grading.add_chop(Chop(**new_args))

        new_grading.inverted = self.inverted
        if invert:
            new_grading.inverted = not new_grading.inverted

        return new_grading

    @property
    def count(self) -> int:
        """Return number of cells, summed over all sub-divisions"""
        return sum(d.count for d in self.chop_data)

    @property
    def is_defined(self) -> bool:
        """Return True is grading is defined;
        It is if there's at least one division added"""
        return len(self.chops) > 0

    @property
    def description(self) -> str:
        """Output string for blockMeshDict"""
        # TODO! Put this into writer
        if not self.is_defined:
            raise UndefinedGradingsError(f"Grading not defined: {self}")

        if len(self.specification) == 1:
            # its a one-number simpleGrading:
            return str(self.specification[0][2])

        # multi-grading: make a nice list
        # FIXME: make a nicer list
        length_ratio_sum = 0.0
        out = "("

        for spec in self.specification:
            out += f"({spec[0]} {spec[1]} {spec[2]})"
            length_ratio_sum += spec[0]

        out += ")"

        if not math.isclose(length_ratio_sum, 1, rel_tol=constants.TOL):
            warnings.warn(f"Length ratio doesn't add up to 1: {length_ratio_sum}", stacklevel=2)

        return out

    def __eq__(self, other_grading):
        # this works theoretically but numerics will probably ruin the party:
        # return self.specification == other.specification
        this_spec = self.get_specification(False)
        other_spec = other_grading.get_specification(False)

        if len(this_spec) != len(other_spec):
            return False

        # so just compare number-by-number
        for i, this in enumerate(this_spec):
            other = other_spec[i]

            for j, this_value in enumerate(this):
                other_value = other[j]

                if not math.isclose(this_value, other_value, rel_tol=constants.TOL):
                    return False

        return True

    def __repr__(self) -> str:
        if self.is_defined:
            return f"Grading ({len(self.chops)} chops {self.description})"

        return f"Grading ({len(self.chops)})"
