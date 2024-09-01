from typing import Dict, List

from classy_blocks.grading.chop import Chop
from classy_blocks.types import AxisType
from classy_blocks.util.frame import Frame


class ChopCollector:
    def __init__(self):
        self.axis_chops: Dict[AxisType, List[Chop]] = {0: [], 1: [], 2: []}
        self.edge_chops = Frame[Chop]()

    def add_axis_chop(self, axis: AxisType, chop: Chop) -> None:
        self.axis_chops[axis].append(chop)

    def add_edge_chop(self, corner_1: int, corner_2: int, chop: Chop) -> None:
        self.edge_chops.add_beam(corner_1, corner_2, chop)

    def clear_axis(self, axis: AxisType) -> None:
        self.axis_chops[axis] = []

    def clear_all(self) -> None:
        self.edge_chops = Frame[Chop]()

    def __getitem__(self, index):
        return self.axis_chops[index]
