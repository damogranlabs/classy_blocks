import abc
from typing import List, Optional, Tuple

from classy_blocks.grading import relations as gr
from classy_blocks.grading.autograding.params.smooth import SmoothGraderParams
from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.chop import Chop
from classy_blocks.util.constants import VBIG


class Layer(abc.ABC):
    """A common interface to all layers of a grading (inflation, buffer, bulk)"""

    # Defines one chop and tools to handle it
    start_size: float
    c2c_expansion: float
    length: float
    count: int
    end_size: float

    def _construct(
        self, length_limit: float = VBIG, count_limit: int = 10**12, size_limit: float = VBIG
    ) -> Tuple[float, float, int]:
        # stop construction of the layer when it hits any of the above limits
        count = 1
        size = self.start_size
        length = self.start_size

        for i in range(count_limit):
            if length >= length_limit:
                break

            if count >= count_limit:
                break

            if size >= size_limit:
                break

            length += size
            size *= self.c2c_expansion
            count = i

        return length, size, count

    def __init__(self, length_limit: float = VBIG, count_limit: int = 10**12, size_limit: float = VBIG):
        # stop construction of the layer when it hits any of the above limits
        self.length, self.end_size, self.count = self._construct(length_limit, count_limit, size_limit)

    def get_chop_count(self, total_count: int) -> int:
        return min(self.count, total_count)

    def get_chop(self, actual_length: float, remaining_count: int, invert: bool) -> Tuple[Chop, int]:
        """Returns a Chop, adapter to a given length; count will not exceed remaining"""
        # length ratios will be normalized later
        length, end_size, count = self._construct(actual_length, remaining_count)
        chop = Chop(length_ratio=length, end_size=end_size, count=count)

        if invert:
            chop.start_size = None
            chop.end_size = self.start_size

        return chop, count

    def __repr__(self):
        return f"{self.length}-{self.count}"


class InflationLayer(Layer):
    def __init__(self, wall_size: float, c2c_expansion: float, thickness_factor: int, _max_length: float):
        self.start_size = wall_size
        self.c2c_expansion = c2c_expansion

        super().__init__(length_limit=thickness_factor * wall_size)


class BufferLayer(Layer):
    def __init__(self, start_size: float, c2c_expansion: float, bulk_size: float, _max_length: float):
        self.start_size = start_size
        self.c2c_expansion = c2c_expansion

        super().__init__(size_limit=bulk_size)

    def get_chop_count(self, total_count: int) -> int:
        # use all remaining cells
        return max(self.count, total_count)


class BulkLayer(Layer):
    # Must be able to adapt end_size to match size_after
    def __init__(self, start_size: float, end_size: float, remainder: float):
        self.start_size = start_size
        self.end_size = end_size
        self.length = remainder

        total_expansion = gr.get_total_expansion__start_size__end_size(self.length, self.start_size, self.end_size)
        self.count = gr.get_count__total_expansion__start_size(self.length, total_expansion, self.start_size)
        self.c2c_expansion = gr.get_c2c_expansion__count__end_size(self.length, self.count, self.end_size)


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

    def add(self, layer: Layer) -> bool:
        self.layers.append(layer)
        return self.is_done

    @property
    def is_done(self) -> bool:
        """Returns True if no more layers need to be added"""
        if len(self.layers) == 0:
            return False

        if len(self.layers) == 3:
            # nothing more to be added?
            return True

        return self.remaining_length <= self.layers[0].start_size

    @property
    def last_size(self) -> float:
        return self.layers[-1].end_size

    def get_chops(self, _total_count: int, _invert: bool) -> List[Chop]:
        # normalize length_ratios
        # ratios = [chop.length_ratio for chop in chops]

        # for chop in chops:
        #     chop.length_ratio = chop.length_ratio / sum(ratios)

        # return chops
        raise NotImplementedError


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

    def get_bulk_layer(self, remaining_length: float, size_after: float) -> BulkLayer:
        return BulkLayer(self.bulk_cell_size, size_after, remaining_length)

    def get_stack(self, length: float, size_after: Optional[float] = None) -> LayerStack:
        stack = LayerStack(length)

        inflation = self.get_inflation_layer(length)
        if stack.add(inflation):
            return stack

        buffer = self.get_buffer_layer(stack.layers[0].end_size, stack.remaining_length)
        if stack.add(buffer):
            return stack

        if size_after is None:
            size_after = self.bulk_cell_size
        bulk = self.get_bulk_layer(stack.remaining_length, size_after)
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

        # TODO: replace 0.9 with something less arbitrary (a better rule)
        return self.get_stack(info.length).last_size < 0.9 * self.bulk_cell_size

    def get_chops(self, count, info: WireInfo) -> List[Chop]:
        if not (info.starts_at_wall or info.ends_at_wall):
            return super().get_chops(count, info)

        stack = self.get_stack(info.length, info.size_after)

        if info.starts_at_wall and info.ends_at_wall:
            raise NotImplementedError

        if info.ends_at_wall:
            return list(reversed(stack.get_chops(count, True)))

        return stack.get_chops(count, False)
