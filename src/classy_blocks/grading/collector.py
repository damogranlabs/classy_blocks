from classy_blocks.grading.chop import Chop


class ChopCollector:
    def __init__(self):
        self.axis_chops: list[Chop] = []
        self.edge_chops: dict[int, list[Chop]] = {
            0: [],
            1: [],
            2: [],
            3: [],
        }

    def chop_axis(self, chop: Chop) -> None:
        self.axis_chops.append(chop)

    def chop_edge(self, i_edge: int, chop: Chop) -> None:
        self.edge_chops[i_edge].append(chop)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.axis_chops)

    def __getitem__(self, i: int):
        return self.axis_chops[i]
