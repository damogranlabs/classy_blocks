from classy_blocks.cbtyping import DirectionType
from classy_blocks.grading.chop import Chop
from classy_blocks.util.frame import Frame


class ChopCollector:
    def _prepare(self) -> None:
        self.axis_chops: dict[DirectionType, list[Chop]] = {0: [], 1: [], 2: []}
        self.edge_chops = Frame[list[Chop]]()

        self._edge_chops_count = 0  # a quick indicator if there are any edge gradings

    def __init__(self) -> None:
        self._prepare()

    def chop_axis(self, direction: DirectionType, chop: Chop) -> None:
        self.axis_chops[direction].append(chop)

    def chop_edge(self, corner_1: int, corner_2: int, chop: Chop) -> None:
        chops = self.edge_chops[corner_1][corner_2]
        if chops is not None:
            chops.append(chop)
        else:
            chops = [chop]

        self.edge_chops.add_beam(corner_1, corner_2, chops)
        self._edge_chops_count += 1

    def clear(self) -> None:
        self._prepare()

    @property
    def is_edge_chopped(self) -> bool:
        return self._edge_chops_count > 0

    def __len__(self):
        return len(self.axis_chops)

    def __getitem__(self, i: DirectionType):
        return self.axis_chops[i]
