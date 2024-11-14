import abc
import dataclasses
import warnings
from typing import List, Optional, Tuple

import scipy.optimize

import classy_blocks.grading.relations as gr
from classy_blocks.grading.autograding.probe import WireInfo
from classy_blocks.grading.chop import Chop

CellSizeType = Optional[float]


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
        """Calculates count based on given length and position"""

    @abc.abstractmethod
    def is_squeezed(self, count: int, info: WireInfo) -> bool:
        """Returns True if cells have to be 'squished' together (thinner than prescribed in params)"""

    @abc.abstractmethod
    def get_chops(self, count: int, info: WireInfo) -> List[Chop]:
        """Fixes cell count but modifies chops so that proper cell sizing will be obeyed"""


@dataclasses.dataclass
class FixedCountParams(ChopParams):
    count: int = 8

    def get_count(self, _length):
        return self.count

    def is_squeezed(self, _count, _info) -> bool:
        return True  # grade everything in first pass

    def get_chops(self, count, _info) -> List[Chop]:
        return [Chop(count=count)]


@dataclasses.dataclass
class SimpleGraderParams(ChopParams):
    cell_size: float

    def get_count(self, length: float):
        return int(length / self.cell_size)

    def is_squeezed(self, _count, _info) -> bool:
        return True

    def get_chops(self, count, _info: WireInfo):
        return [Chop(count=count)]


@dataclasses.dataclass
class SmoothGraderParams(ChopParams):
    cell_size: float

    def get_count(self, length: float):
        # the first chop defines the count;
        count = int(length / self.cell_size)
        # it must be divisible by 2
        if count % 2 != 0:
            count += 1

        return count

    def is_squeezed(self, count, info) -> bool:
        return info.length <= self.cell_size * count

    def define_sizes(self, size_before: CellSizeType, size_after: CellSizeType) -> Tuple[float, float]:
        """Defines start and end cell size.
        size_before and size_after are taken from preceding/following wires;
        when a size is None, this is the last/first wire."""
        if size_before == 0 or size_after == 0:
            # until all counts/sizes are defined
            # (the first pass with uniform grading),
            # there's no point in doing anything
            raise RuntimeError("Undefined grading encountered!")

        if size_before is None:
            size_before = self.cell_size

        if size_after is None:
            size_after = self.cell_size

        return size_before, size_after

    def get_chops(self, count, info):
        halfcount = count // 2

        size_before, size_after = self.define_sizes(info.size_before, info.size_after)

        # choose length ratio so that cells at the middle of blocks
        # (between the two chops) have the same size
        def fobj(lratio):
            chop_1 = Chop(length_ratio=lratio, count=halfcount, start_size=size_before)
            data_1 = chop_1.calculate(info.length)

            chop_2 = Chop(length_ratio=1 - lratio, count=halfcount, end_size=size_after)
            data_2 = chop_2.calculate(info.length)

            ratio = (data_1.end_size - data_2.start_size) ** 2

            return ratio, [chop_1, chop_2]

        # it's not terribly important to minimize until the last dx
        tol = min(size_before, size_after, self.cell_size) * 0.1
        results = scipy.optimize.minimize_scalar(lambda r: fobj(r)[0], bounds=[0.1, 0.9], options={"xatol": tol})
        if not results.success:  # type:ignore
            warnings.warn("Could not determine optimal grading", stacklevel=1)

        return fobj(results.x)[1]  # type:ignore


# INVALID! Next on list
@dataclasses.dataclass
class InflationGraderParams(ChopParams):
    """See description of InflationGrader"""

    first_cell_size: float
    bulk_cell_size: float

    c2c_expansion: float = 1.2
    bl_thickness_factor: int = 30
    buffer_expansion: float = 2

    def is_squeezed(self, _count: int, _info: WireInfo) -> bool:
        return False

    @property
    def inflation_layer_thickness(self) -> float:
        return self.first_cell_size * self.bl_thickness_factor

    def _get_inflation_chop(self, length: float) -> Tuple[Chop, float]:
        """Creates a Chop for the inflation layer; returns size of the last cell"""
        near_wall = Chop(
            length_ratio=self.inflation_layer_thickness / length,
            start_size=self.first_cell_size,
            c2c_expansion=self.c2c_expansion,
        )
        data = near_wall.calculate(length)
        return (near_wall, data.end_size)

    def _get_buffer_chop(self, start_size: float) -> Tuple[Chop, float]:
        """Creates a chop between the last cell of inflation layer
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
        chops: List[Chop] = []

        if length < self.inflation_layer_thickness:
            warnings.warn("Inflation layer is thicker than block size!", stacklevel=1)

        # near-wall sizes:
        near_wall, last_bl_size = self._get_inflation_chop(length)
        remaining_length = length - self.inflation_layer_thickness
        chops.append(near_wall)

        if remaining_length <= 0:
            warnings.warn("Stopping chops at inflation layer (not enough space)!", stacklevel=1)
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
