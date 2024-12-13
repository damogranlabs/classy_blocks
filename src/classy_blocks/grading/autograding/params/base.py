import abc
from typing import List, Optional

from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.chop import Chop

CellSizeType = Optional[float]


def sum_length(start_size: float, count: int, c2c_expansion: float) -> float:
    """Returns absolute length of the chop"""
    length = 0.0
    size = start_size

    for _ in range(count):
        length += size
        size *= c2c_expansion

    return length


class ChopParams(abc.ABC):
    @abc.abstractmethod
    def get_count(self, length: float, start_at_wall: bool, end_at_wall: bool) -> int:
        """Calculates count based on given length and position"""

    @abc.abstractmethod
    def is_squeezed(self, count: int, info: WireInfo) -> bool:
        """Returns True if cells have to be 'squished' together (thinner than prescribed in params)"""

    def get_squeezed_chops(self, count: int, _info: WireInfo) -> List[Chop]:
        """Different chopping rules for squeezed blocks"""
        return [Chop(count=count)]

    @abc.abstractmethod
    def get_chops(self, count: int, info: WireInfo) -> List[Chop]:
        """Fixes cell count but modifies chops so that proper cell sizing will be obeyed"""
