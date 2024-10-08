import abc
import dataclasses
import warnings
from typing import List, Tuple

import scipy.optimize

import classy_blocks.grading.relations as gr
from classy_blocks.grading.chop import Chop
from classy_blocks.types import ChopTakeType


def sum_length(start_size: float, count: int, c2c_expansion: float) -> float:
    """Returns absolute length of the chop"""
    length = 0.0
    size = start_size

    for _ in range(count):
        length += size
        size *= c2c_expansion

    return length


class ChopParams(abc.ABC):
    @abc.abstractmethod
    def get_count(self, length: float) -> int:
        """Calculates count based on given length - used once only"""

    @abc.abstractmethod
    def get_chops(self, count: int, length: float, size_before: float = 0, size_after: float = 0) -> List[Chop]:
        """Fixes cell count but modifies chops so that proper cell sizing will be obeyed"""
        # That depends on inherited classes' philosophy


@dataclasses.dataclass
class FixedCountParams(ChopParams):
    count: int = 8

    def get_count(self, _length):
        return self.count

    def get_chops(self, count, _length, _size_before=0, _size_after=0) -> List[Chop]:
        return [Chop(count=count)]


# TODO: rename this CentipedeCaseClassNameMonstrosity
@dataclasses.dataclass
class SimpleChopParams(ChopParams):
    cell_size: float

    def get_count(self, length: float):
        return int(length / self.cell_size)

    def get_chops(self, count, _length, _size_before=0, _size_after=0):
        return [Chop(count=count)]


@dataclasses.dataclass
class HighReChopParams(ChopParams):
    cell_size: float

    def get_count(self, length: float):
        # the first chop defines the count;
        count = int(length / self.cell_size)
        # it must be divisible by 2
        if count % 2 != 0:
            count += 1

        return count

    def get_chops(self, count, length, size_before=0, size_after=0):
        # length of the wire that was used to set count
        if size_before == 0:
            size_before = self.cell_size
        if size_after == 0:
            size_after = self.cell_size

        chops = [
            Chop(count=count // 2),
            Chop(count=count // 2),
        ]

        def objfun(params):
            chops[0].length_ratio = params[0]
            chops[1].length_ratio = 1 - params[0]

            chops[0].total_expansion = params[1]
            chops[1].total_expansion = params[2]

            data_1 = chops[0].calculate(length)
            data_2 = chops[1].calculate(length)

            ofstart = (size_before - data_1.start_size) ** 2
            ofmid1 = (data_1.end_size - self.cell_size) ** 2
            ofmid2 = (data_2.start_size - self.cell_size) ** 2
            ofend = (data_2.end_size - size_after) ** 2

            return max([ofstart, ofmid1, ofmid2, ofend])

        initial = [0.5, 1, 1]
        bounds = (
            (0.1, 0.9),
            (0.1, 10),
            (0.1, 10),
        )
        result = scipy.optimize.minimize(objfun, initial, bounds=bounds).x

        chops[0].length_ratio = result[0]
        chops[1].length_ratio = 1 - result[0]

        chops[0].total_expansion = result[1]
        chops[1].total_expansion = result[2]

        return chops


# INVALID! Next on list
@dataclasses.dataclass
class LowReChopParams(ChopParams):
    """Parameters for mesh grading for Low-Re cases.
    To save on cell count, only a required thickness (boundary layer)
    will be covered with thin cells (c2c_expansion in size ratio between them).
    Then a bigger expansion ratio will be applied between the last cell of boundary layer
    and the first cell of the bulk flow.

    Example:
     ________________
    |
    |                 > bulk size (cell_size=bulk, no expansion)
    |________________
    |
    |________________ > buffer layer (c2c = 2)
    |________________
    |================ > boundary layer (cell_size=y+, c2c=1.2)
    / / / / / / / / / wall

    Args:
        first_cell_size (float): thickness of the first cell near the wall
        c2c_expansion (float): expansion ratio between cells in boundary layer
        bl_thickness_factor (int): thickness of the boundary layer in y+ units (relative to first_cell_size)
        buffer_expansion (float): expansion between cells in buffer layer
        bulk_cell_size (float): size of cells inside the domain

        Autochop will take all relevant blocks and choose one to start with - set cell counts
        and other parameters that must stay fixed for all further blocks.
        It will choose the longest/shortest ('max/min') block edge or something in between ('avg').
        The finest grid will be obtained with 'max', the coarsest with 'min'.
    """

    first_cell_size: float
    bulk_cell_size: float

    c2c_expansion: float = 1.2
    bl_thickness_factor: int = 30
    buffer_expansion: float = 2

    take: ChopTakeType = "avg"

    @property
    def boundary_layer_thickness(self) -> float:
        return self.first_cell_size * self.bl_thickness_factor

    def _get_boundary_chop(self, length: float) -> Tuple[Chop, float]:
        """Creates a Chop for the boundary layer; returns size of the last cell"""
        near_wall = Chop(
            length_ratio=self.boundary_layer_thickness / length,
            start_size=self.first_cell_size,
            c2c_expansion=self.c2c_expansion,
        )
        data = near_wall.calculate(length)
        return (near_wall, data.end_size)

    def _get_buffer_chop(self, start_size: float) -> Tuple[Chop, float]:
        """Creates a chop between the last cell of boundary layer
        and the first cell of bulk flow; returns length of the chop"""
        buffer_count = gr.get_count__total_expansion__c2c_expansion(
            1, self.bulk_cell_size / start_size, self.buffer_expansion
        )
        buffer_size = sum_length(start_size, buffer_count, self.buffer_expansion)
        buffer = Chop(start_size=start_size, c2c_expansion=self.buffer_expansion, count=buffer_count)

        return buffer, buffer_size

    def _get_bulk_chop(self, remaining_size: float) -> Chop:
        count = max(1, int(remaining_size / self.bulk_cell_size))
        return Chop(count=count)

    def get_count(self, length: float):
        # TODO! Do
        chops: List[Chop] = []

        if length < self.boundary_layer_thickness:
            warnings.warn("Boundary layer is thicker than block size!", stacklevel=1)

        # near-wall sizes:
        near_wall, last_bl_size = self._get_boundary_chop(length)
        remaining_length = length - self.boundary_layer_thickness
        chops.append(near_wall)

        if remaining_length <= 0:
            warnings.warn("Stopping chops at boundary layer (not enough space)!", stacklevel=1)
            # return chops
            return 0

        # buffer
        buffer, buffer_size = self._get_buffer_chop(last_bl_size)
        buffer.length_ratio = buffer_size / length
        chops.append(buffer)
        if buffer_size >= remaining_length:
            warnings.warn("Stopping chops at buffer layer (not enough space)!", stacklevel=1)
            # return chops
            return 1

        # bulk
        remaining_length = remaining_length - buffer_size
        bulk = self._get_bulk_chop(remaining_length)
        bulk.length_ratio = remaining_length / length
        chops.append(bulk)

        # return chops
        return 1

    def get_chops(self, count, length, size_before=0, size_after=0) -> List[Chop]:
        raise NotImplementedError("TODO!")
