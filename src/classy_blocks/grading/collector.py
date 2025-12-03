from classy_blocks.cbtyping import DirectionType
from classy_blocks.grading.chop import Chop, EdgeChop
from classy_blocks.util.frame import Frame


class ChopCollector:
    def _prepare(self) -> None:
        self.axis_chops: dict[DirectionType, list[Chop]] = {0: [], 1: [], 2: []}
        self.edge_chops = Frame[list[EdgeChop]]()

    def __init__(self) -> None:
        self._prepare()

    def chop_axis(self, axis: DirectionType, chop: Chop) -> None:
        self.axis_chops[axis].append(chop)

    def chop_edge(self, corner_1: int, corner_2: int, chop: Chop) -> None:
        chops = self.edge_chops[corner_1][corner_2]
        chops.append(chop)

    def clear(self) -> None:
        self._prepare()

    def __len__(self):
        return len(self.axis_chops)

    def __getitem__(self, i: DirectionType):
        return self.axis_chops[i]
