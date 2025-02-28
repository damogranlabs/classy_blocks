"""The Mesh object ties everything together and writes the blockMeshDict in the end."""

from typing import List, Optional, Set, Union, get_args

from classy_blocks.base.exceptions import EdgeNotFoundError
from classy_blocks.cbtyping import DirectionType
from classy_blocks.construct.assemblies.assembly import Assembly
from classy_blocks.construct.operations.operation import Operation
from classy_blocks.construct.shape import Shape
from classy_blocks.construct.stack import Stack
from classy_blocks.items.block import Block
from classy_blocks.items.patch import Patch
from classy_blocks.items.vertex import Vertex
from classy_blocks.lists.block_list import BlockList
from classy_blocks.lists.edge_list import EdgeList
from classy_blocks.lists.face_list import FaceList
from classy_blocks.lists.geometry_list import GeometryList
from classy_blocks.lists.patch_list import PatchList
from classy_blocks.lists.vertex_list import VertexList
from classy_blocks.util import constants
from classy_blocks.util.vtk_writer import write_vtk

AdditiveType = Union[Operation, Shape, Stack, Assembly]


class Mesh:
    """contains blocks, edges and all necessary methods for assembling blockMeshDict"""

    def __init__(self) -> None:
        # List of all added operations/shapes
        self.depot: List[AdditiveType] = []
        self.deleted: Set[Operation] = set()

        self.vertex_list = VertexList()
        self.edge_list = EdgeList()
        self.block_list = BlockList()
        self.patch_list = PatchList()
        self.face_list = FaceList()
        self.geometry_list = GeometryList()

        self.settings = {
            "prescale": None,
            "scale": 1,
            "transform": None,
            "mergeType": None,  # use 'points' to fall back to the older point-based block merging
            "checkFaceCorrespondence": None,  # true by default, turn off if blockMesh fails (3-sided pyramids etc.)
            "verbose": None,
        }

    def add(self, entity: AdditiveType) -> None:
        """Add a classy_blocks entity to the mesh (Operation or a Shape)"""
        # this does nothing yet;
        # the data will be processed automatically at an
        # appropriate occasion (before write/optimize)
        self.depot.append(entity)

    def _add_vertices(self, operation: Operation) -> List[Vertex]:
        """Creates/finds vertices from operation's points and returns them"""
        vertices: List[Vertex] = []

        # FIXME: prettify/move logic elsewhere/remove private method
        for corner in range(8):
            point = operation.points[corner]
            # remove master patches, only slave will remain
            patches = operation.get_patches_at_corner(corner)
            patches = patches.intersection(self.patch_list.slave_patches)
            new_vertices = self.vertex_list.add(point, list(patches))
            vertices.append(new_vertices)

        return vertices

    def merge_patches(self, master: str, slave: str) -> None:
        """Merges two non-conforming named patches using face merging;
        https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility#x13-470004.3.2
        (breaks the 100% hex-mesh rule)"""
        self.patch_list.merge(master, slave)

    def set_default_patch(self, name: str, kind: str) -> None:
        """Adds the 'defaultPatch' entry to the mesh; any non-specified block boundaries
        will be assigned this patch"""
        self.patch_list.set_default(name, kind)

    def modify_patch(self, name: str, kind: str, settings: Optional[List[str]] = None) -> None:
        """Fetches a patch named 'patch' and modifies its type and optionally
        other settings. They are passed on to blockMeshDict as a list of strings
        as-is, with no additional brain power used"""
        self.patch_list.modify(name, kind, settings)

    def add_geometry(self, geometry: dict) -> None:
        """Adds named entry in the 'geometry' section of blockMeshDict;
        'geometry' is in the form of dictionary {'geometry_name': [list of properties]};
        properties are as specified by searchable* class in documentation.
        See examples/advanced/project for an example."""
        self.geometry_list.add(geometry)

    def delete(self, operation: Operation) -> None:
        """Excludes the given operation from any processing;
        the data remains but it will not contribute to the mesh"""
        self.deleted.add(operation)

    @property
    def _operations(self) -> List[Operation]:
        operations: List[Operation] = []

        for entity in self.depot:
            if isinstance(entity, Operation):
                ops_to_add = [entity]
            else:
                ops_to_add = entity.operations

            for op in ops_to_add:
                if op not in self.deleted:
                    operations.append(op)

        return operations

    def _add_geometry(self) -> None:
        for entity in self.depot:
            if entity.geometry is not None:
                self.add_geometry(entity.geometry)

    def assemble(self) -> None:
        """Converts classy_blocks entities (operations and shapes) to
        actual vertices, edges, blocks and other stuff to be inserted into
        blockMeshDict. After this has been done, the above objects
        cease to have any function or influence on mesh."""
        if self.is_assembled:
            # don't assemble twice but do update wire lengths
            self.block_list.update()
            return

        operations = self._operations
        op_vertices: List[List[Vertex]] = []

        for operation in operations:
            # create vertices and edges
            vertices = self._add_vertices(operation)
            self.edge_list.add_from_operation(vertices, operation)
            op_vertices.append(vertices)  # blocks will be created from those

            # get patches and faces
            self.patch_list.add(vertices, operation)
            self.face_list.add(vertices, operation)

        # then, create blocks from known vertices and edges
        for i, operation in enumerate(operations):
            block = Block(len(self.block_list.blocks), op_vertices[i])
            for wire in block.wire_list:
                try:
                    edge = self.edge_list.find(*wire.vertices)
                    block.add_edge(wire.corners[0], wire.corners[1], edge)
                except EdgeNotFoundError:
                    continue

            for direction in get_args(DirectionType):
                block.add_chops(direction, operation.chops[direction])

            block.cell_zone = operation.cell_zone

            self.block_list.add(block)

        self._add_geometry()
        self.block_list.update()

    def clear(self) -> None:
        """Undoes the assemble() method; clears created blocks and other lists
        but leaves added depot items intact"""
        self.vertex_list.clear()
        self.edge_list.clear()
        self.block_list.clear()
        self.patch_list.clear()
        self.face_list.clear()

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

    def format_settings(self) -> str:
        """Put self.settings in a proper, blockMesh-readable format"""
        out = ""

        for key, value in self.settings.items():
            if value is not None:
                out += f"{key} {value};\n"

        out += "\n"

        return out

    def write(self, output_path: str, debug_path: Optional[str] = None) -> None:
        """Writes a blockMeshDict to specified location. If debug_path is specified,
        a VTK file is created first where each block is a single cell, to see simplified
        blocking in case blockMesh fails with an unfriendly error message."""
        if not self.is_assembled:
            self.assemble()

        if debug_path is not None:
            write_vtk(debug_path, self.vertex_list.vertices, self.block_list.blocks)

        # gradings: define after writing VTK;
        # if it is not specified correctly, this will raise an exception
        self.block_list.assemble()

        with open(output_path, "w", encoding="utf-8") as output:
            output.write(constants.MESH_HEADER)

            output.write(self.format_settings())

            output.write(self.geometry_list.description)

            output.write(self.vertex_list.description)
            output.write(self.block_list.description)
            output.write(self.edge_list.description)
            output.write(self.face_list.description)
            output.write(self.patch_list.description)

            output.write(constants.MESH_FOOTER)

    @property
    def is_assembled(self) -> bool:
        """Returns True if assemble() has been executed on this mesh"""
        return len(self.vertex_list.vertices) > 0

    @property
    def vertices(self) -> List[Vertex]:
        return self.vertex_list.vertices

    @property
    def patches(self) -> List[Patch]:
        return list(self.patch_list.patches.values())

    @property
    def operations(self) -> List[Operation]:
        """Returns a list of operations from all entities in depot"""
        operations: List[Operation] = []

        for entity in self.depot:
            if isinstance(entity, Operation):
                operations.append(entity)
            else:
                operations += entity.operations

        return operations

    @property
    def blocks(self) -> List[Block]:
        return self.block_list.blocks
