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

import dataclasses
import math
import warnings
from typing import List

from classy_blocks.grading.chop import Chop
from classy_blocks.types import GradingSpecType
from classy_blocks.util import constants


class Grading:
    """Grading specification for a single edge"""

    def __init__(self, length: float):
        # "multi-grading" specification according to:
        # https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading
        self.length = length  # to be updated when adding/modifying block edges
        self.chops: List[Chop] = []

    @property
    def specification(self) -> GradingSpecType:
        # a list of lists [length ratio, count ratio, total expansion]
        spec = []

        for chop in self.chops:
            if not (0 < chop.length_ratio <= 1):
                raise ValueError(f"Length ratio must be between 0 and (including) 1, got {chop.length_ratio}")

            length = self.length * chop.length_ratio
            chop_data = chop.calculate(length)
            spec.append([chop.length_ratio, chop_data.count, chop_data.total_expansion])

        return spec

    def add_chop(self, chop: Chop) -> None:
        self.chops.append(chop)

    def copy(self, length: float, invert: bool) -> "Grading":
        """Creates a new grading with the same chops (counts) on a different length"""
        new_grading = Grading(length)

        for chop in self.chops:
            # calculate chops on current grading to get the correct counts
            old_data = chop.calculate(self.length)
            # create a copy of this Chop with equal count but
            # set other parameters from current data so that
            # the correct start/end size or c2c is maintained"""
            new_args = dataclasses.asdict(old_data)
            new_args["count"] = old_data.count

            for arg in ["total_expansion", "c2c_expansion", "start_size", "end_size"]:
                new_args[arg] = None

            new_args[old_data.preserve] = dataclasses.asdict(old_data)[old_data.preserve]

            chop = Chop(**new_args)
            if invert:
                chop.invert()

            new_grading.add_chop(chop)

        return new_grading

    @property
    def counts(self) -> List[int]:
        """Counts per chop"""
        return [int(d[1]) for d in self.specification]

    @property
    def count(self) -> int:
        """Return number of cells, summed over all sub-divisions"""
        return sum(self.counts)

    @property
    def is_defined(self) -> bool:
        """Return True is grading is defined;
        It is if there's at least one division added"""
        return len(self.chops) > 0

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
            warnings.warn(f"Length ratio doesn't add up to 1: {length_ratio_sum}", stacklevel=2)

        return out

    def __eq__(self, other):
        # this works theoretically but numerics will probably ruin the party:
        # return self.specification == other.specification
        if len(self.specification) != len(other.specification):
            return False

        # so just compare number-by-number
        for i, this_spec in enumerate(self.specification):
            other_spec = other.specification[i]

            for j, this_value in enumerate(this_spec):
                other_value = other_spec[j]

                if not math.isclose(this_value, other_value, rel_tol=constants.TOL):
                    return False

        return True

    def __repr__(self) -> str:
        return str(self.specification)
