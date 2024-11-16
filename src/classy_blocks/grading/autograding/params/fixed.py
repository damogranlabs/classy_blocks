import dataclasses
from typing import List

from classy_blocks.grading.autograding.params.base import ChopParams
from classy_blocks.grading.chop import Chop


@dataclasses.dataclass
class FixedCountGraderParams(ChopParams):
    count: int = 8

    def get_count(self, _length, _start_at_wall, _end_at_wall):
        return self.count

    def is_squeezed(self, _count, _info) -> bool:
        return True  # grade everything in first pass

    def get_chops(self, count, _info) -> List[Chop]:
        return [Chop(count=count)]
