from typing import List

from classy_blocks.items.block import Block
from classy_blocks.items.vertex import Vertex


def write_vtk(path: str, vertices: List[Vertex], blocks: List[Block]) -> None:
    """Generates a simple VTK file where each block is a hexahedral cell;
    useful for debugging blockMesh's FATAL_ERRORs"""
    # A sample VTK file with all cell types; only hexahedrons are used (cell type 12)
    # vtk DataFile Version 2.0
    # classy_blocks debug output
    # ASCII

    # DATASET UNSTRUCTURED_GRID
    # POINTS 27 float
    # 0 0 0    1 0 0    2 0 0    0 1 0    1 1 0    2 1 0
    # 0 0 1    1 0 1    2 0 1    0 1 1    1 1 1    2 1 1
    # 0 1 2    1 1 2    2 1 2    0 1 3    1 1 3    2 1 3
    # 0 1 4    1 1 4    2 1 4    0 1 5    1 1 5    2 1 5
    # 0 1 6    1 1 6    2 1 6

    # CELLS 11 60

    # 8 0 1 4 3 6 7 10 9
    # 8 1 2 5 4 7 8 11 10
    # 4 6 10 9 12
    # 4 5 11 10 14
    # 6 15 16 17 14 13 12
    # 6 18 15 19 16 20 17
    # 4 22 23 20 19
    # 3 21 22 18
    # 3 22 19 18
    # 2 26 25
    # 1 24

    # CELL_TYPES 11
    # 12
    # 12
    # 10
    # 10
    # 7
    # 6
    # 9
    # 5
    # 5
    # 3
    # 1

    # CELL_DATA 11
    # SCALARS block_ids float 1
    # LOOKUP_TABLE default
    # 0
    # 1
    # 2
    # 3
    # 4
    # 5
    # 6
    # 7
    # 8
    # 9
    # 10

    with open(path, "w", encoding="utf-8") as output:
        n_blocks = len(blocks)

        header = "# vtk DataFile Version 2.0\n" + "classy_blocks debug output\n" + "ASCII\n"

        output.write(header)

        # points
        output.write("\nDATASET UNSTRUCTURED_GRID\n")
        output.write(f"POINTS {len(vertices)} float\n")

        for vertex in vertices:
            output.write(f"{vertex.position[0]} {vertex.position[1]} {vertex.position[2]}\n")

        # cells
        output.write(f"\nCELLS {n_blocks} {9*n_blocks}\n")

        for block in blocks:
            output.write("8")
            for vertex in block.vertices:
                output.write(f" {vertex.index}")
            output.write("\n")

        # cell types
        output.write(f"\nCELL_TYPES {n_blocks}\n")
        for _ in blocks:
            output.write("12\n")

        # cell data
        output.write(f"\nCELL_DATA {n_blocks}\n")
        output.write("SCALARS block_ids float 1\n")
        output.write("LOOKUP_TABLE default\n")

        for i in range(n_blocks):
            output.write(f"{i}\n")
