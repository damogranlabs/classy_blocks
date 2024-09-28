import functools
from typing import Set, get_args

from classy_blocks.items.block import Block
from classy_blocks.items.wires.axis import Axis
from classy_blocks.mesh import Mesh
from classy_blocks.types import AxisType


@functools.lru_cache(maxsize=3000)  # that's for 1000 blocks
def _get_block_from_axis(mesh: Mesh, axis: Axis) -> Block:
    for block in mesh.blocks:
        for index in get_args(AxisType):
            if block.axes[index] == axis:
                return block

    raise RuntimeError("Block for Axis not found!")


class Probe:
    """Examines the mesh and gathers required data for auto chopping"""

    def __init__(self, mesh: Mesh):
        self.mesh = mesh

    def _get_block_from_axis(self, axis: Axis) -> Block:
        return _get_block_from_axis(self.mesh, axis)

    def get_blocks_on_layer(self, block: Block, axis: AxisType) -> Set[Block]:
        """Returns all blocks on the same 'layer' as the one in arguments"""
        # blocks to be returned
        blocks: Set[Block] = set()
        # blocks not to check again
        traversed: Set[Block] = set()

        def check(blk: Block):
            if blk not in traversed:
                traversed.add(blk)

                for neighbour_axis in blk.axes[axis].neighbours:
                    neighbour_block = self._get_block_from_axis(neighbour_axis)
                    blocks.add(neighbour_block)

                    check(self._get_block_from_axis(neighbour_axis))

        check(block)

        return blocks
