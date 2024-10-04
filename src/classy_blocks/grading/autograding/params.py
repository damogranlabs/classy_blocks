import abc
import dataclasses
import warnings
from typing import List, Tuple

from classy_blocks.grading import relations as gr
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
    def get_chops_from_length(self, length: float) -> List[Chop]:
        """Calculates chopping parameters based on given length"""

    @abc.abstractmethod
    def adjust_chops(self, count: int, length: float) -> List[Chop]:
        """Leaves cell count unchanged but modifies chops
        so that proper cell sizing will be obeyed"""
        # That depends on inherited classes' philosophy


@dataclasses.dataclass
class FixedCountParams(ChopParams):
    count: int = 8

    def get_chops_from_length(self, _length):
        return [Chop(count=self.count, total_expansion=1)]

    def adjust_chops(self, _count, _length) -> List[Chop]:
        return self.get_chops_from_length(0)


# TODO: rename this CentipedeCaseClassNameMonstrosity
@dataclasses.dataclass
class SimpleChopParams(ChopParams):
    cell_size: float

    def get_chops_from_length(self, length: float) -> List[Chop]:
        return [Chop(count=int(length / self.cell_size), total_expansion=1)]

    def adjust_chops(self, count, _length) -> List[Chop]:
        return [Chop(count=count)]


@dataclasses.dataclass
class HighReChopParams(ChopParams):
    """Parameters for mesh grading for high-Re cases.
    Two chops are added to all blocks; c2c_expansion and and length_ratio
    are utilized to keep cell sizes between blocks consistent"""

    def get_chops_from_length(self, length: float) -> List[Chop]:
        raise NotImplementedError

    @abc.abstractmethod
    def adjust_chops(self, count: int, length: float) -> List[Chop]:
        raise NotImplementedError


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

    def get_chops_from_length(self, length: float) -> List[Chop]:
        chops: List[Chop] = []

        if length < self.boundary_layer_thickness:
            warnings.warn("Boundary layer is thicker than block size!", stacklevel=1)

        # near-wall sizes:
        near_wall, last_bl_size = self._get_boundary_chop(length)
        remaining_length = length - self.boundary_layer_thickness
        chops.append(near_wall)

        if remaining_length <= 0:
            warnings.warn("Stopping chops at boundary layer (not enough space)!", stacklevel=1)
            return chops

        # buffer
        buffer, buffer_size = self._get_buffer_chop(last_bl_size)
        buffer.length_ratio = buffer_size / length
        chops.append(buffer)
        if buffer_size >= remaining_length:
            warnings.warn("Stopping chops at buffer layer (not enough space)!", stacklevel=1)
            return chops

        # bulk
        remaining_length = remaining_length - buffer_size
        bulk = self._get_bulk_chop(remaining_length)
        bulk.length_ratio = remaining_length / length
        chops.append(bulk)

        return chops

    def adjust_chops(self, count: int, length: float) -> List[Chop]:
        raise NotImplementedError("TODO!")
