import dataclasses
import warnings
from typing import Tuple

import scipy.optimize

from classy_blocks.grading.autograding.params.base import CellSizeType, ChopParams
from classy_blocks.grading.chop import Chop


@dataclasses.dataclass
class SmoothGraderParams(ChopParams):
    cell_size: float

    def get_count(self, length: float, _start_at_wall, _end_at_wall):
        # the first chop defines the count;
        count = int(length / self.cell_size)
        # it must be divisible by 2
        if count % 2 != 0:
            count += 1

        return count

    def is_squeezed(self, count, info) -> bool:
        return info.length <= self.cell_size * count

    def define_sizes(self, size_before: CellSizeType, size_after: CellSizeType) -> Tuple[float, float]:
        """Defines start and end cell size.
        size_before and size_after are taken from preceding/following wires;
        when a size is None, this is the last/first wire."""
        if size_before == 0 or size_after == 0:
            # until all counts/sizes are defined
            # (the first pass with uniform grading),
            # there's no point in doing anything
            raise RuntimeError("Undefined grading encountered!")

        if size_before is None:
            size_before = self.cell_size

        if size_after is None:
            size_after = self.cell_size

        return size_before, size_after

    def get_chops(self, count, info):
        halfcount = count // 2

        size_before, size_after = self.define_sizes(info.size_before, info.size_after)

        # choose length ratio so that cells at the middle of blocks
        # (between the two chops) have the same size
        def fobj(lratio):
            chop_1 = Chop(length_ratio=lratio, count=halfcount, start_size=size_before)
            data_1 = chop_1.calculate(info.length)

            chop_2 = Chop(length_ratio=1 - lratio, count=halfcount, end_size=size_after)
            data_2 = chop_2.calculate(info.length)

            ratio = (data_1.end_size - data_2.start_size) ** 2

            return ratio, [chop_1, chop_2]

        # it's not terribly important to minimize until the last dx
        tol = min(size_before, size_after, self.cell_size) * 0.1
        results = scipy.optimize.minimize_scalar(lambda r: fobj(r)[0], bounds=[0.1, 0.9], options={"xatol": tol})
        if not results.success:  # type:ignore
            warnings.warn("Could not determine optimal grading", stacklevel=1)

        return fobj(results.x)[1]  # type:ignore
