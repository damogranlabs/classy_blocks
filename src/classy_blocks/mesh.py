from typing import Optional, Union

from classy_blocks.assemble.assembler import MeshAssembler
from classy_blocks.assemble.depot import Depot
from classy_blocks.assemble.dump import AssembledDump, DumpBase, EmptyDump
from classy_blocks.assemble.settings import Settings
from classy_blocks.cbtyping import GeometryType
from classy_blocks.construct.assemblies.assembly import Assembly
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import Shape
from classy_blocks.construct.stack import Stack
from classy_blocks.items.block import Block
from classy_blocks.items.patch import Patch
from classy_blocks.items.vertex import Vertex
from classy_blocks.util.constants import TOL
from classy_blocks.write.vtk import write_vtk
from classy_blocks.write.writer import MeshWriter

AdditiveType = Union[Operation, Shape, Stack, Assembly]


class Mesh:
    """contains blocks, edges and all necessary methods for assembling blockMeshDict"""

    def __init__(self) -> None:
        # List of all added/deleted operations/shapes
        self.depot = Depot()
        self.settings = Settings()

        # container for items - assembled blocks, vertices, patches, etc.
        self.dump: DumpBase = EmptyDump()

    def add(self, solid: AdditiveType) -> None:
        """Add a classy_blocks solid to the mesh (Loft, Shape, Assembly, ...)"""
        # this does nothing yet;
        # the data will be processed automatically at an
        # appropriate occasion (before write/optimize)
        self.depot.add_solid(solid)

    def merge_patches(self, master: str, slave: str) -> None:
        """Merges two non-conforming named patches using face merging;
        https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility#x13-470004.3.2
        (breaks the 100% hex-mesh rule)"""
        self.settings.merged_patches.append((master, slave))

    def set_default_patch(self, name: str, kind: str) -> None:
        """Adds the 'defaultPatch' entry to the mesh; any non-specified block boundaries
        will be assigned this patch"""
        self.settings.default_patch = {"name": name, "kind": kind}

    def modify_patch(self, name: str, kind: str, settings: Optional[list[str]] = None) -> None:
        """Fetches a patch named 'patch' and modifies its type and optionally
        other settings. They are passed on to blockMeshDict as a list of strings
        as-is, with no additional brain power used"""
        self.settings.modify_patch(name, kind, settings)

    def add_geometry(self, geometry: GeometryType) -> None:
        """Adds named entry in the 'geometry' section of blockMeshDict;
        'geometry' is in the form of dictionary {'geometry_name': [list of properties]};
        properties are as specified by searchable* class in documentation.
        See examples/advanced/project for an example."""
        self.settings.add_geometry(geometry)

    def delete(self, operation: Operation) -> None:
        """Excludes the given operation from any processing;
        the data remains but it will not contribute to the mesh"""
        self.depot.delete_solid(operation)

    def assemble(self, merge_tol: float = TOL) -> AssembledDump:
        """Converts classy_blocks entities (operations and shapes) to
        actual vertices, edges, blocks and other stuff to be inserted into
        blockMeshDict. After this has been done, the above objects
        cease to have any function or influence on mesh."""
        if self.is_assembled:
            assert isinstance(self.dump, AssembledDump)
            return self.dump

        assembler = MeshAssembler(self.depot, self.settings, merge_tol)
        self.dump = assembler.assemble()
        return self.dump

    def clear(self) -> None:
        """Undoes the assemble() method; clears created blocks and other lists
        but leaves added depot items intact"""
        self.dump = EmptyDump()

    def backport(self) -> None:
        """When mesh is assembled, points from depot are converted to vertices and
        operations are converted to blocks. When vertices are edited (modification/optimization),
        depot entities remain unchanged. This can cause problems with some edges
        (Origin, Axis, ...) and future stuff.

        This method updates depot from blocks, clears all lists and reassembles the
        mesh as if it was modified from the start."""
        if not self.is_assembled:
            raise RuntimeError("Cannot backport non-assembled mesh")

        operations = self.operations
        blocks = self.blocks

        for i, block in enumerate(blocks):
            op = operations[i]

            vertices = [vertex.position for vertex in block.vertices]
            op.bottom_face.update(vertices[:4])
            op.top_face.update(vertices[4:])

        self.clear()
        self.assemble()

    def write(self, output_path: str, debug_path: Optional[str] = None, merge_tol: float = TOL) -> None:
        """Writes a blockMeshDict to specified location. If debug_path is specified,
        a VTK file is created first where each block is a single cell, to see simplified
        blocking in case blockMesh fails with an unfriendly error message."""
        if not self.is_assembled:
            self.assemble(merge_tol)

        if debug_path is not None:
            write_vtk(debug_path, self.vertices, self.blocks)

        # gradings: define after writing VTK;
        # if it is not specified correctly, this will raise an exception
        self.dump.finalize()

        assert isinstance(self.dump, AssembledDump)  # to pacify type checker
        writer = MeshWriter(self.dump, self.settings)
        writer.write(output_path)

    @property
    def is_assembled(self) -> bool:
        """Returns True if assemble() has been executed on this mesh"""
        return self.dump.is_assembled

    @property
    def vertices(self) -> list[Vertex]:
        return self.dump.vertices

    @property
    def patches(self) -> list[Patch]:
        return list(self.dump.patches)

    @property
    def operations(self) -> list[Operation]:
        """Returns a list of operations from all entities in depot"""
        return self.depot.operations

    @property
    def blocks(self) -> list[Block]:
        return self.dump.blocks
