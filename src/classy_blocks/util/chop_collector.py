from typing import Dict, get_args

from classy_blocks.grading.chop import Chop
from classy_blocks.grading.grading import Grading
from classy_blocks.types import AxisType
from classy_blocks.util import constants
from classy_blocks.util.frame import Frame


class ChopCollector:
    """Creates Gradings and handles adding Chops to it"""

    # TODO: TEST

    def __init__(self):
        self.axis_gradings: Dict[AxisType, Grading] = {0: Grading(0), 1: Grading(0), 2: Grading(0)}
        self.edge_gradings = Frame[Grading]()

        self._create_beams()

    def _create_beams(self) -> None:
        # create all Gradings in Frame in advance
        for pair in constants.EDGE_PAIRS:
            self.edge_gradings.add_beam(pair[0], pair[1], Grading(0))

    def add_axis_chop(self, axis: AxisType, chop: Chop) -> None:
        self.axis_gradings[axis].add_chop(chop)

    def add_edge_chop(self, corner_1: int, corner_2: int, chop: Chop) -> None:
        self.edge_gradings[corner_1][corner_2].add_chop(chop)

    def clear_axis(self, axis: AxisType) -> None:
        self.axis_gradings[axis].chops = []

    def clear_all(self) -> None:
        for axis in get_args(AxisType):
            self.axis_gradings[axis] = Grading(0)

        self._create_beams()
        self._create_beams()
