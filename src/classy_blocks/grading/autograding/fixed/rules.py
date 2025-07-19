import dataclasses

from classy_blocks.grading.autograding.rules import ChopRules
from classy_blocks.grading.chop import Chop


@dataclasses.dataclass
class FixedCountRules(ChopRules):
    count: int = 8

    def get_count(self, _length, _start_at_wall, _end_at_wall):
        return self.count

    def is_squeezed(self, _count, _info) -> bool:
        return True  # grade everything in first pass

    def get_chops(self, _count, _info) -> list[Chop]:
        # In FixedCountGrader this is never called
        # (everything is graded as 'squeezed')
        raise RuntimeError
