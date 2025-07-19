import dataclasses

from classy_blocks.cbtyping import ChopTakeType, DirectionType
from classy_blocks.items.block import Block
from classy_blocks.items.wires.axis import Axis
from classy_blocks.items.wires.wire import Wire


@dataclasses.dataclass
class Entry:
    block: Block
    # blocks of different orientations can belong to the same row;
    # remember how they are oriented
    heading: DirectionType
    # also keep track of blocks that are upside-down;
    # 'True' means the block is overturned
    flipped: bool

    @property
    def axis(self) -> Axis:
        return self.block.axes[self.heading]

    @property
    def wires(self) -> list[Wire]:
        return self.axis.wires.wires

    @property
    def neighbours(self) -> set[Axis]:
        return self.axis.neighbours

    @property
    def lengths(self) -> list[float]:
        return [wire.length for wire in self.wires]


class Row:
    """A string of blocks that must share the same count
    because they sit next to each other.

    This may not be an actual 'string' of blocks because,
    depending on blocking, a whole 'layer' of blocks can be
    chopped by specifying a single block only (for example, direction 2 in a cylinder)"""

    def __init__(self) -> None:
        self.entries: list[Entry] = []

        # the whole row must share the same cell count;
        # it's determined if it's greater than 0
        self.count = 0

    def add_block(self, block: Block, heading: DirectionType) -> None:
        axis = block.axes[heading]
        axis.grade()

        # check neighbours for alignment
        if len(self.entries) == 0:
            flipped = False
        else:
            # find a block's neighbour among existing entries
            # and determine its relative alignment
            for entry in self.entries:
                if axis in entry.neighbours:
                    flipped = not entry.axis.is_aligned(axis)
                    if entry.flipped:
                        flipped = not flipped

                    break
            else:
                # TODO: nicer message
                raise RuntimeError("No neighbour found!")

        self.entries.append(Entry(block, heading, flipped))

        # take count from block, if it's manually defined
        # TODO: make this a separate method
        # TODO: un-ififif by handling axis' chops separately
        for wire in axis.wires:
            if wire.is_defined:
                if self.count != 0:
                    if self.count != wire.grading.count:
                        # TODO! Custom exception
                        raise RuntimeError(
                            f"Inconsistent counts (existing {self.count}, replaced with {wire.grading.count}, "
                            f"block {block.index} direction {axis.direction})"
                        )

                self.count = wire.grading.count

    def get_length(self, take: ChopTakeType = "avg"):
        lengths: list[float] = []
        for entry in self.entries:
            lengths += entry.lengths

        if take == "min":
            return min(lengths)

        if take == "max":
            return max(lengths)

        return sum(lengths) / len(self.entries) / 4  # "avg"

    def get_axes(self) -> list[Axis]:
        return [entry.axis for entry in self.entries]

    def get_wires(self) -> list[Wire]:
        wires: list[Wire] = []

        for axis in self.get_axes():
            wires += axis.wires

        return wires

    @property
    def blocks(self) -> list[Block]:
        return [entry.block for entry in self.entries]
