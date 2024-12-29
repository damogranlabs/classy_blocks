import abc
from typing import List, Tuple

from classy_blocks.grading import relations as gr
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

        for _ in range(count_limit):
            if length >= length_limit:
                break

            if count >= count_limit:
                break

            if size >= size_limit:
                break

            length += size
            size *= self.c2c_expansion
            count += 1

        return length, size, count

    def __init__(self, length_limit: float = VBIG, count_limit: int = 10**12, size_limit: float = VBIG):
        # stop construction of the layer when it hits any of the above limits
        self.length, self.end_size, self.count = self._construct(length_limit, count_limit, size_limit)


class InflationLayer(Layer):
    def __init__(self, wall_size: float, c2c_expansion: float, thickness_factor: int, max_length: float):
        self.start_size = wall_size
        self.c2c_expansion = c2c_expansion

        super().__init__(length_limit=min(thickness_factor * wall_size, max_length))


class BufferLayer(Layer):
    def __init__(self, start_size: float, c2c_expansion: float, bulk_size: float, _max_length: float):
        self.start_size = start_size
        self.c2c_expansion = c2c_expansion

        super().__init__(size_limit=bulk_size)


class BulkLayer(Layer):
    # Must be able to adapt end_size to match size_after
    def __init__(self, start_size: float, end_size: float, remainder: float):
        self.start_size = start_size
        self.end_size = end_size
        self.length = remainder

        # TODO: handle this in a more dignified way
        if remainder < min(start_size, end_size):
            self.count = 0
            self.c2c_expansion = 1
        else:
            total_expansion = gr.get_total_expansion__start_size__end_size(self.length, self.start_size, self.end_size)
            self.count = gr.get_count__total_expansion__start_size(self.length, total_expansion, self.start_size)

            if self.count < 2:
                self.c2c_expansion = 1
            else:
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
        if layer.count > 0:
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
