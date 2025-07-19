import functools
from typing import get_args

from classy_blocks.base.exceptions import BlockNotFoundError, NoInstructionError
from classy_blocks.cbtyping import DirectionType
from classy_blocks.grading.autograding.row import Row
from classy_blocks.items.block import Block
from classy_blocks.items.wires.axis import Axis
from classy_blocks.mesh import Mesh


@functools.lru_cache(maxsize=3000)  # that's for 1000 blocks
def get_block_from_axis(mesh: Mesh, axis: Axis) -> Block:
    for block in mesh.blocks:
        if axis in block.axes:
            return block

    raise RuntimeError("Block for Axis not found!")


class Instruction:
    """A descriptor that tells in which direction the specific block can be chopped."""

    def __init__(self, block: Block):
        self.block = block
        self.directions: list[bool] = [False] * 3

    @property
    def is_defined(self):
        return all(self.directions)

    def __hash__(self) -> int:
        return id(self)


class RowCatalogue:
    """A collection of rows on a specified axis"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

        self.rows: dict[DirectionType, list[Row]] = {0: [], 1: [], 2: []}
        self.instructions = [Instruction(block) for block in mesh.blocks]

        for i in get_args(DirectionType):
            self._populate(i)

    def _get_undefined_instructions(self, direction: DirectionType) -> list[Instruction]:
        return [i for i in self.instructions if not i.directions[direction]]

    def _find_instruction(self, block: Block):
        # TODO: perform dedumbing on this exquisite piece of code
        for instruction in self.instructions:
            if instruction.block == block:
                return instruction

        raise NoInstructionError(f"No instruction found for block {block}")

    def _add_block_to_row(self, row: Row, instruction: Instruction, direction: DirectionType) -> None:
        row.add_block(instruction.block, direction)
        instruction.directions[direction] = True

        block = instruction.block

        for neighbour_axis in block.axes[direction].neighbours:
            neighbour_block = get_block_from_axis(self.mesh, neighbour_axis)

            if neighbour_block in row.blocks:
                continue

            instruction = self._find_instruction(neighbour_block)

            self._add_block_to_row(row, instruction, neighbour_block.get_axis_direction(neighbour_axis))

    def _populate(self, direction: DirectionType) -> None:
        while True:
            undefined_instructions = self._get_undefined_instructions(direction)
            if len(undefined_instructions) == 0:
                break

            row = Row()
            self._add_block_to_row(row, undefined_instructions[0], direction)
            self.rows[direction].append(row)

    def get_row_blocks(self, block: Block, direction: DirectionType) -> list[Block]:
        for row in self.rows[direction]:
            if block in row.blocks:
                return row.blocks

        raise BlockNotFoundError(f"Direction {direction} of {block} not in catalogue")
