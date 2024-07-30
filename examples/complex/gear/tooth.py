from typing import ClassVar

import classy_blocks as cb


class ToothSketch(cb.MappedSketch):
    quads: ClassVar = [
        # layers on tooth wall
        [0, 1, 18, 17],  # 0
        [1, 2, 19, 18],  # 1
        [2, 3, 20, 19],  # 2
        [3, 4, 21, 20],  # 3
        [4, 5, 22, 21],  # 4
        [5, 6, 23, 22],  # 5
        [6, 7, 8, 23],  # 6
        # surrounding blocks
        [8, 9, 22, 23],  # 7
        [9, 10, 21, 22],  # 8
        [11, 12, 21, 10],  # 9
        [12, 13, 20, 21],  # 10
        [13, 14, 15, 20],  # 11
        [15, 16, 19, 20],  # 12
        [16, 17, 18, 19],  # 13
    ]

    # In x-direction, only one half needs to be chopped, the other half will be
    # copied from teeth on the other side.

    # In y-direction, block 10 should also be in the list
    # except that it doesn't need boundary layers that will be created for blocks on the tooth.
    # It will be chopped manually.
    chops: ClassVar = [
        [1, 2, 9, 10, 11],
        [0],
    ]

    def __init__(self, positions, curve: cb.LinearInterpolatedCurve):
        self.curve = curve
        super().__init__(positions, self.quads)

    def add_edges(self) -> None:
        for i in range(7):
            # If all edges refer to the same curve, it will be
            # transformed N-times, N being the number of entities
            # where that curve is used. Therefore copy it
            # so that each edge will have its own object to work with.
            self.faces[i].add_edge(0, cb.OnCurve(self.curve.copy()))

        for i in (9, 10, 11):
            self.faces[i].add_edge(0, cb.Origin([0, 0, 0]))
