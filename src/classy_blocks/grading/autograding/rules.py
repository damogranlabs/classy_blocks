import abc

from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.chop import Chop


class ChopRules(abc.ABC):
    """Provides  information and mechanisms for determining cell count and
    grading for autograders."""

    @abc.abstractmethod
    def get_count(self, length: float, start_at_wall: bool, end_at_wall: bool) -> int:
        """Calculates count based on given length and position"""

    @abc.abstractmethod
    def is_squeezed(self, count: int, info: WireInfo) -> bool:
        """Returns True if cells have to be 'squished' together (thinner than prescribed in params)"""

    def get_squeezed_chops(self, count: int, _info: WireInfo) -> list[Chop]:
        """Different chopping rules for squeezed blocks"""
        return [Chop(count=count)]

    @abc.abstractmethod
    def get_chops(self, count: int, info: WireInfo) -> list[Chop]:
        """Fixes cell count but modifies chops so that proper cell sizing will be obeyed"""
