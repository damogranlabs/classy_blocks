import dataclasses

from classy_blocks.grading import relations as gr


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
            1, self.bulk_cell_size / self.inflation_end_size, self.buffer_expansion
        )

    @property
    def buffer_thickness(self) -> float:
        return sum_length(self.inflation_end_size, self.buffer_count, self.buffer_expansion)
