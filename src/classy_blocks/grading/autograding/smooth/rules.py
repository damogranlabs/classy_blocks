import dataclasses
import warnings

import scipy.optimize

from classy_blocks.cbtyping import CellSizeType
from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.autograding.rules import ChopRules
from classy_blocks.grading.chop import Chop


def distribute_cells(count, length, size_before, size_after) -> list[Chop]:
    # TODO: put back in SmoothRules
    if length < max(size_before, size_after):
        return [Chop(count=count)]

    count_1 = count // 2
    count_2 = count - count_1

    # choose length ratio so that cells at the middle of blocks
    # (between the two chops) have the same size
    def fobj(lratio):
        chop_1 = Chop(length_ratio=lratio, count=count_1, start_size=size_before)
        data_1 = chop_1.calculate(length)

        chop_2 = Chop(length_ratio=1 - lratio, count=count_2, end_size=size_after)
        data_2 = chop_2.calculate(length)

        ratio = (data_1.end_size - data_2.start_size) ** 2

        return ratio, [chop_1, chop_2]

    # it's not terribly important to minimize until the last dx
    tol = min(size_before, size_after) * 0.01
    try:
        results = scipy.optimize.minimize_scalar(lambda r: fobj(r)[0], bounds=[0.1, 0.9], options={"xatol": tol})
    except ValueError:  # TODO: custom exception for grading relations
        return [Chop(count=count)]

    if not results.success:  # type:ignore
        warnings.warn("Could not determine optimal grading", stacklevel=1)

    return fobj(results.x)[1]  # type:ignore


@dataclasses.dataclass
class SmoothRules(ChopRules):
    cell_size: float

    def get_count(self, length: float, _start_at_wall, _end_at_wall):
        # the first chop defines the count;
        count = int(length / self.cell_size)

        # can't use zero
        if count == 0:
            count = 2

        return count

    def is_squeezed(self, count, info) -> bool:
        return info.length < self.cell_size * count

    def define_sizes(self, size_before: CellSizeType, size_after: CellSizeType) -> tuple[float, float]:
        """Defines start and end cell size.
        size_before and size_after are taken from preceding/following wires;
        when a size is None, this is the last/first wire."""
        if size_before is None:
            size_before = self.cell_size

        if size_after is None:
            size_after = self.cell_size

        return size_before, size_after

    def get_squeezed_chops(self, count: int, _info: WireInfo) -> list[Chop]:
        return [Chop(count=count)]

    def get_chops(self, count, info):
        size_before, size_after = self.define_sizes(info.size_before, info.size_after)
        return distribute_cells(count, info.length, size_before, size_after)
