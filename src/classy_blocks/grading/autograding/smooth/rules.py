import dataclasses
from typing import List, Tuple

from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.autograding.rules import ChopRules
from classy_blocks.grading.autograding.smooth.distributor import SmoothDistributor
from classy_blocks.grading.chop import Chop
from classy_blocks.types import CellSizeType


@dataclasses.dataclass
class SmoothRules(ChopRules):
    cell_size: float

    def get_count(self, length: float, _start_at_wall, _end_at_wall):
        # the first chop defines the count;
        count = int(length / self.cell_size)
        # it must be divisible by 2
        if count % 2 != 0:
            count += 1

        # can't use zero
        if count == 0:
            count = 2

        return count

    def is_squeezed(self, count, info) -> bool:
        return info.length < self.cell_size * count

    def define_sizes(self, size_before: CellSizeType, size_after: CellSizeType) -> Tuple[float, float]:
        """Defines start and end cell size.
        size_before and size_after are taken from preceding/following wires;
        when a size is None, this is the last/first wire."""
        if size_before == 0 or size_after == 0:
            # until all counts/sizes are defined
            # (the first pass with uniform grading),
            # there's no point in doing anything
            # TODO: check the same with other graders
            raise RuntimeError("Undefined grading encountered!")

        if size_before is None:
            size_before = self.cell_size

        if size_after is None:
            size_after = self.cell_size

        return size_before, size_after

    def get_squeezed_chops(self, count: int, _info: WireInfo) -> List[Chop]:
        return [Chop(count=count)]

    def get_chops(self, count, info):
        size_before, size_after = self.define_sizes(info.size_before, info.size_after)

        smoother = SmoothDistributor(count, size_before, info.length, size_after)

        return smoother.get_chops(2)
