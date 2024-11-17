import abc
from typing import List

import classy_blocks.grading.relations as gr
from classy_blocks.grading.autograding.params.base import sum_length
from classy_blocks.grading.autograding.params.smooth import SmoothGraderParams
from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.chop import Chop


class Layer(abc.ABC):
    """A common interface to all layers of a grading (inflation, buffer, bulk)"""

    # Defines one chop and tools to handle it
    start_size: float
    c2c_expansion: float

    def __init__(self, remainder: float):
        # remaining length of the wire
        self.remainder = remainder

    @property
    @abc.abstractmethod
    def length(self) -> float:
        """Returns overall length of this layer"""

    @property
    def count(self) -> int:
        """Returns cell count in this layer"""
        length = min(self.length, self.remainder)
        return gr.get_count__start_size__c2c_expansion(length, self.start_size, self.c2c_expansion)

    @property
    def end_size(self) -> float:
        """Size of the last cell in this layer"""
        return self.start_size * self.c2c_expansion**self.count

    @property
    def is_final(self) -> bool:
        """Returns True if this layer is the last (no more space for additional ones)"""
        return self.length >= self.remainder

    def get_chop(self, total_count: int, invert: bool) -> Chop:
        """Returns a Chop with either this layer's count or given one,
        whichever is lower"""
        # length ratios will be normalized later
        if invert:
            return Chop(
                length_ratio=self.length,
                end_size=self.end_size,
                c2c_expansion=1 / self.c2c_expansion,
                count=min(self.count, total_count),
            )

        return Chop(
            length_ratio=self.length,
            start_size=self.start_size,
            c2c_expansion=self.c2c_expansion,
            count=min(self.count, total_count),
        )

    def __repr__(self):
        return f"{self.length}-{self.count}"


class InflationLayer(Layer):
    def __init__(self, wall_size: float, c2c_expansion: float, thickness_factor: int, max_length: float):
        self.start_size = wall_size
        self.c2c_expansion = c2c_expansion
        self.thickness_factor = thickness_factor

        super().__init__(max_length)

    @property
    def length(self):
        return min(self.remainder, self.start_size * self.thickness_factor)


class BufferLayer(Layer):
    def __init__(self, start_size: float, c2c_expansion: float, bulk_size: float, max_length: float):
        self.start_size = start_size  # *c2c_expansion (the first cell is already bigger?)
        self.c2c_expansion = c2c_expansion
        self.bulk_size = bulk_size

        self.total_expansion = self.bulk_size / self.start_size

        # manually sum up those few cells that lead from start to bulk size
        count = 0
        size = self.start_size
        length = 0.0

        while size <= self.bulk_size:
            length += size
            count += 1

            if length > max_length:
                break

            size *= self.c2c_expansion

        self._count = count
        self._last_size = size

        super().__init__(max_length)

    @property
    def count(self):
        return self._count

    @property
    def length(self):
        return sum_length(self.start_size, self.count, self.c2c_expansion)

    @property
    def last_size(self):
        return self._last_size


class BulkLayer(Layer):
    def __init__(self, cell_size: float, remainder: float):
        self.start_size = cell_size

        self.cell_size = cell_size
        self.c2c_expansion = 1

        super().__init__(remainder)

    @property
    def length(self):
        return self.remainder

    @property
    def last_size(self):
        return self.cell_size


class LayerStack:
    """A collection of one, two or three layers (chops) for InflationGrader"""

    def __init__(self, length: float):
        self.length = length
        self.layers: List[Layer] = []

    @property
    def count(self) -> int:
        return sum(layer.count for layer in self.layers)

    @property
    def remaining_length(self) -> float:
        return self.length - sum(layer.length for layer in self.layers)

    def add(self, layer: Layer) -> None:
        """Returns True when the added layer was the final one"""
        self.layers.append(layer)

    @property
    def is_done(self) -> bool:
        """Returns True if no more layers need to be added"""
        if len(self.layers) == 3:
            # nothing more to be added?
            return True

        return self.remaining_length <= 0

    def get_chops(self, total_count: int, invert: bool) -> List[Chop]:
        chops: List[Chop] = []

        for layer in self.layers:
            chop = layer.get_chop(total_count, invert)
            chops.append(chop)

            total_count -= layer.count

            if total_count <= 0:
                break

        # normalize length_ratios
        ratios = [chop.length_ratio for chop in chops]

        for chop in chops:
            chop.length_ratio = chop.length_ratio / sum(ratios)

        return chops


class InflationGraderParams(SmoothGraderParams):
    """See description of InflationGrader"""

    # TODO: refactor to a reasonable number of 'if' clauses

    def __init__(
        self,
        first_cell_size: float,
        bulk_cell_size: float,
        c2c_expansion: float = 1.2,
        bl_thickness_factor: int = 30,
        buffer_expansion: float = 2,
    ):
        self.first_cell_size = first_cell_size
        self.bulk_cell_size = bulk_cell_size
        self.c2c_expansion = c2c_expansion
        self.bl_thickness_factor = bl_thickness_factor
        self.buffer_expansion = buffer_expansion

        # use SmoothGrader's logic for bulk chops
        self.cell_size = self.bulk_cell_size

    def get_inflation_layer(self, max_length: float) -> InflationLayer:
        return InflationLayer(self.first_cell_size, self.c2c_expansion, self.bl_thickness_factor, max_length)

    def get_buffer_layer(self, start_size, max_length: float) -> BufferLayer:
        return BufferLayer(start_size, self.buffer_expansion, self.bulk_cell_size, max_length)

    def get_bulk_layer(self, remaining_length: float) -> BulkLayer:
        return BulkLayer(self.bulk_cell_size, remaining_length)

    def get_stack(self, length: float) -> LayerStack:
        stack = LayerStack(length)

        inflation = self.get_inflation_layer(length)
        stack.add(inflation)
        if stack.is_done:
            return stack

        buffer = self.get_buffer_layer(stack.layers[0].end_size, stack.remaining_length)
        stack.add(buffer)
        if stack.is_done:
            return stack

        bulk = self.get_bulk_layer(stack.remaining_length)
        stack.add(bulk)

        return stack

    def get_count(self, length: float, starts_at_wall: bool, ends_at_wall: bool):
        if not (starts_at_wall or ends_at_wall):
            return super().get_count(length, False, False)

        if starts_at_wall and ends_at_wall:
            # this will produce 1 extra chop (the middle one could be
            # common to both bulk chops) but it doesn't matter at this moment
            stack = self.get_stack(length / 2)
            return 2 * stack.count

        stack = self.get_stack(length)
        return stack.count

    def is_squeezed(self, count: int, info: WireInfo) -> bool:
        if not (info.starts_at_wall or info.ends_at_wall):
            return super().is_squeezed(count, info)

        # a squeezed wire is one that can't fit all layers
        # or one that can't fit all cells
        stack = self.get_stack(info.length)

        if len(stack.layers) == 3:
            return stack.count < count

        return True

    def get_chops(self, count, info: WireInfo) -> List[Chop]:
        if not (info.starts_at_wall or info.ends_at_wall):
            return super().get_chops(count, info)

        stack = self.get_stack(info.length)

        if info.starts_at_wall and info.ends_at_wall:
            raise NotImplementedError

        if info.ends_at_wall:
            return list(reversed(stack.get_chops(count, True)))

        return stack.get_chops(count, False)
