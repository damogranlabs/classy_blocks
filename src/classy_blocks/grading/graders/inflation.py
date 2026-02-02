import abc
import copy
import dataclasses
from typing import Union

from classy_blocks.assemble.dump import AssembledDump
from classy_blocks.cbtyping import ChopTakeType, DirectionType
from classy_blocks.grading.analyze.row import Row
from classy_blocks.grading.define import relations as gr
from classy_blocks.grading.define.chop import Chop
from classy_blocks.grading.graders.auto import AutoGraderMixin
from classy_blocks.grading.graders.fixed import FixedAxisGrader
from classy_blocks.grading.graders.manager import AxisGrader, GradingManager
from classy_blocks.items.block import Block
from classy_blocks.mesh import Mesh
from classy_blocks.util.constants import VBIG


def sum_length(first_cell_size: float, count: int, c2c_expansion: float) -> float:
    length: float = 0
    size: float = first_cell_size

    for _ in range(count):
        length += size
        size *= c2c_expansion

    return length


@dataclasses.dataclass
class InflationParams:
    """Common parameters for inflation grading (see inflation/grader)"""

    first_cell_size: float
    bulk_cell_size: float

    c2c_expansion: float = 1.2
    bl_thickness_factor: int = 30
    buffer_expansion: float = 2

    @property
    def bl_thickness(self) -> float:
        return self.first_cell_size * self.bl_thickness_factor

    @property
    def inflation_count(self) -> int:
        """Number of cells in inflation layer"""
        return gr.get_count__start_size__c2c_expansion(self.bl_thickness, self.first_cell_size, self.c2c_expansion)

    @property
    def inflation_end_size(self) -> float:
        """Size of the last cell of inflation layer"""
        total_expansion = gr.get_total_expansion__count__c2c_expansion(
            self.bl_thickness, self.inflation_count, self.c2c_expansion
        )
        return self.first_cell_size * total_expansion

    @property
    def buffer_start_size(self) -> float:
        """Size of the first cell in the buffer layer"""
        return self.inflation_end_size * self.buffer_expansion

    @property
    def buffer_count(self) -> int:
        return gr.get_count__total_expansion__c2c_expansion(
            1, self.bulk_cell_size / self.buffer_start_size, self.buffer_expansion
        )

    @property
    def buffer_thickness(self) -> float:
        return sum_length(self.buffer_start_size, self.buffer_count, self.buffer_expansion)


class Layer(abc.ABC):
    """A common interface to all layers of a grading (inflation, buffer, bulk)"""

    # Defines one chop and tools to handle it
    start_size: float
    c2c_expansion: float
    length: float
    count: int
    end_size: float

    length_ratio: float = 1

    def _construct(
        self, length_limit: float = VBIG, count_limit: int = 10**12, size_limit: float = VBIG
    ) -> tuple[float, float, int]:
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

    def __init__(
        self, params: InflationParams, length_limit: float = VBIG, count_limit: int = 10**12, size_limit: float = VBIG
    ):
        self.params = params
        # stop construction of the layer when it hits any of the above limits
        self.length, self.end_size, self.count = self._construct(length_limit, count_limit, size_limit)

    @abc.abstractmethod
    def get_chop(self) -> Chop:
        # layers define chops differently
        pass

    def invert(self) -> "Layer":
        # inverts the data for a chop
        self.end_size, self.start_size = self.start_size, self.end_size
        self.c2c_expansion = 1 / self.c2c_expansion

        return self

    def copy(self) -> "Layer":
        return copy.deepcopy(self)

    def __repr__(self):
        data = {
            "start_size": self.start_size,
            "c2c_expansion": self.c2c_expansion,
            "length": self.length,
            "count": self.count,
            "end_size": self.end_size,
            "length_ratio": self.length_ratio,
        }
        return str(data)


class InflationLayer(Layer):
    def __init__(self, params: InflationParams, remaining_length: float):
        self.start_size = params.first_cell_size
        self.c2c_expansion = params.c2c_expansion
        super().__init__(params, length_limit=min(params.bl_thickness, remaining_length))

    def get_chop(self):
        return Chop(
            length_ratio=self.length_ratio,
            start_size=self.start_size,
            c2c_expansion=self.c2c_expansion,
        )


class BufferLayer(Layer):
    def __init__(self, params: InflationParams, remaining_length: float):
        self.start_size = params.buffer_start_size
        self.c2c_expansion = params.buffer_expansion
        super().__init__(params, size_limit=params.bulk_cell_size, length_limit=remaining_length)

    def get_chop(self):
        return Chop(length_ratio=self.length_ratio, count=self.count, c2c_expansion=self.c2c_expansion)


class BulkLayer(Layer):
    def __init__(self, params: InflationParams, remaining_length: float):
        self.start_size = params.bulk_cell_size
        self.c2c_expansion = 1
        super().__init__(params, length_limit=remaining_length)

    def get_chop(self):
        return Chop(length_ratio=self.length_ratio, count=self.count)


class LayerStack:
    """A collection of one, two or three layers (chops) for InflationGrader"""

    def _normalize_ratios(self):
        # re-assigns length_ratio to all layers
        sum_lengths = sum(layer.length for layer in self.layers)

        for layer in self.layers:
            layer.length_ratio = layer.length / sum_lengths

    def _construct(self) -> list[Layer]:
        layers: list[Layer] = []
        remaining_length = self.total_length

        inflation_layer = InflationLayer(self.params, remaining_length)
        remaining_length -= inflation_layer.length
        layers.append(inflation_layer)

        if remaining_length < self.params.buffer_start_size:
            return layers

        buffer_layer = BufferLayer(self.params, remaining_length)
        remaining_length -= buffer_layer.length
        layers.append(buffer_layer)

        if remaining_length < self.params.bulk_cell_size:
            return layers

        bulk_layer = BulkLayer(self.params, remaining_length)
        layers.append(bulk_layer)

        return layers

    def __init__(self, params: InflationParams, total_length: float):
        self.params = params
        self.total_length = total_length

        self.layers = self._construct()
        self._normalize_ratios()

    def invert(self) -> "LayerStack":
        for layer in self.layers:
            layer.invert()

        self.layers.reverse()

        return self

    def mirror(self) -> "LayerStack":
        # duplicates layers and inverts the other half;
        # must be done on half of ref_length
        for layer in reversed(self.layers):
            self.layers.append(layer.copy().invert())

        self._normalize_ratios()

        return self

    @property
    def count(self) -> int:
        return sum(layer.count for layer in self.layers)

    @property
    def remaining_length(self) -> float:
        return self.total_length - sum(layer.length for layer in self.layers)


class InflationAxisGrader(AxisGrader):
    def __init__(self, block: Block, direction: DirectionType, params: InflationParams, ref_length: float):
        self.params = params
        self.ref_length = ref_length
        super().__init__(block, direction)

    def get_stack(self) -> LayerStack:
        return LayerStack(self.params, self.ref_length)

    def get_chops(self) -> list[Chop]:
        stack = self.get_stack()
        return [layer.get_chop() for layer in stack.layers]


class InvertedInflationAxisGrader(InflationAxisGrader):
    def get_stack(self):
        stack = super().get_stack()
        stack.invert()

        return stack


class DoubleInflationAxisGrader(InflationAxisGrader):
    def get_stack(self) -> LayerStack:
        stack = LayerStack(self.params, self.ref_length / 2)
        stack.mirror()

        return stack


class BulkAxisGrader(FixedAxisGrader):
    def __init__(self, block: Block, direction: DirectionType, params: InflationParams, ref_length: float):
        super().__init__(block, direction, max(1, int(ref_length / params.bulk_cell_size)))


class InflationGrader(GradingManager, AutoGraderMixin):
    """Parameters for mesh grading for Low-Re cases.
    To save on cell count, only a required thickness (inflation layer)
    will be covered with thin cells (c2c_expansion in size ratio between them).
    Then a bigger expansion ratio will be applied between the last cell of inflation layer
    and the first cell of the bulk flow.

    Example:
     ________________
    |
    |                 > bulk size (cell_size=bulk, no expansion)
    |________________
    |
    |________________ > buffer layer (c2c = 2)
    |________________
    |================ > inflation layer (cell_size=y+, c2c=1.2)
    / / / / / / / / / wall

    Args:
        first_cell_size (float): thickness of the first cell near the wall
        c2c_expansion (float): expansion ratio between cells in inflation layer
        bl_thickness_factor (int): thickness of the inflation layer in y+ units (relative to first_cell_size)
        buffer_expansion (float): expansion between cells in buffer layer
        bulk_cell_size (float): size of cells inside the domain

        The grader will take all relevant blocks and choose one to start with - set cell counts
        and other parameters that must stay fixed for all further blocks.
        It will choose the longest/shortest ('max/min') block edge or something in between ('avg').
        The finest grid will be obtained with 'max', the coarsest with 'min'.
    """

    def __init__(
        self,
        mesh: Mesh,
        first_cell_size: float,
        bulk_cell_size: float,
        c2c_expansion: float = 1.2,
        bl_thickness_factor: int = 30,
        buffer_expansion: float = 2,
        take: ChopTakeType = "avg",
    ):
        self.params = InflationParams(
            first_cell_size, bulk_cell_size, c2c_expansion, bl_thickness_factor, buffer_expansion
        )
        self.take = take

        mesh.assemble()

        assert isinstance(mesh.dump, AssembledDump)

        super().__init__(mesh.dump, mesh.settings)

    def _get_grader(self, row: Row) -> Union[type[InflationAxisGrader], type[BulkAxisGrader]]:
        # use simple grader's method for rows that don't touch walls
        # but a more sophisticated method for rows on-the-wall
        starts_at_wall = False
        ends_at_wall = False

        for entry in row.entries:
            for wire in entry.wires:
                wire_info = self.probe.get_wire_info(wire)
                if wire_info.starts_at_wall:
                    if not entry.flipped:
                        starts_at_wall = True
                if wire_info.ends_at_wall:
                    if not entry.flipped:
                        ends_at_wall = True

                if starts_at_wall and ends_at_wall:
                    print(f"Starts and ends: {wire_info.wire.vertices[0].index, wire_info.wire.vertices[1].index}")
                    return DoubleInflationAxisGrader

        if starts_at_wall:
            return InflationAxisGrader

        if ends_at_wall:
            return InvertedInflationAxisGrader

        return BulkAxisGrader

    def _grade_row(self, row: Row):
        # obey user-specified gradings first
        super()._grade_row(row)

        if row.count == 0:
            # add automatic gradings to non-specified rows
            entry = row.entries[0]
            row_length = self._get_row_length(row)
            axis_grader = self._get_grader(row)(entry.block, entry.heading, self.params, row_length)

            axis_grader.grade()
