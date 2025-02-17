import dataclasses

from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.autograding.rules import ChopRules
from classy_blocks.grading.chop import Chop


@dataclasses.dataclass
class SimpleRules(ChopRules):
    cell_size: float

    def get_count(self, length: float, _start_at_wall, _end_at_wall):
        return int(length / self.cell_size)

    def is_squeezed(self, _count, _info) -> bool:
        return True

    def get_chops(self, count, _info: WireInfo):
        return [Chop(count=count)]
