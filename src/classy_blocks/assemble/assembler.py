from typing import get_args

from classy_blocks.assemble.depot import Depot
from classy_blocks.assemble.dump import AssembledDump
from classy_blocks.assemble.settings import Settings
from classy_blocks.base.exceptions import EdgeNotFoundError
from classy_blocks.cbtyping import DirectionType
from classy_blocks.items.block import Block
from classy_blocks.items.vertex import Vertex
from classy_blocks.lists.block_list import BlockList
from classy_blocks.lists.edge_list import EdgeList
from classy_blocks.lists.face_list import FaceList
from classy_blocks.lists.patch_list import PatchList
from classy_blocks.lists.vertex_list import VertexList
from classy_blocks.lookup.point_registry import HexPointRegistry
from classy_blocks.util import constants


class MeshAssembler:
    def __init__(self, depot: Depot, settings: Settings, merge_tol=constants.TOL):
        self.depot = depot
        self.settings = settings
        self.merge_tol = merge_tol

        # once the mesh is assembled, adding new stuff to depot will break things;
        # better (and faster) is to cache status quo
        self._operations = self.depot.operations

        self._points = HexPointRegistry.from_operations(self._operations, self.merge_tol)

    def _create_blocks(self, vertex_list: VertexList) -> BlockList:
        block_list = BlockList()

        for iop, operation in enumerate(self._operations):
            op_indexes = self._points.cell_addressing[iop]
            op_vertices = [vertex_list.vertices[i] for i in op_indexes]

            # duplicate vertices on slave patches
            for corner in range(8):
                point = operation.points[corner]
                # remove master patches, only slave will remain
                patches = operation.get_patches_at_corner(corner)
                patches = patches.intersection(self.settings.slave_patches)
                if len(patches) == 0:
                    continue

                op_vertices[corner] = vertex_list.add_duplicated(point, patches)

            block = Block(iop, op_vertices)

            for direction in get_args(DirectionType):
                block.add_chops(direction, operation.chops[direction])

            block.cell_zone = operation.cell_zone
            block.visible = operation not in self.depot.deleted

            block_list.add(block)

        return block_list

    def _create_edges(self, block_list: BlockList) -> EdgeList:
        edge_list = EdgeList()

        # skim edges from operations
        for iop, operation in enumerate(self._operations):
            block = block_list.blocks[iop]

            vertices = block.vertices

            for ipnt, point in enumerate(operation.points):
                vertices[ipnt].project(point.projected_to)
                edge_list.add_from_operation(vertices, operation)

        # and add them to blocks
        for iop in range(len(self._operations)):
            block = block_list.blocks[iop]

            for wire in block.wire_list:
                try:
                    edge = edge_list.find(*wire.vertices)
                    block.add_edge(wire.corners[0], wire.corners[1], edge)
                except EdgeNotFoundError:
                    continue

        return edge_list

    def _add_geometry(self):
        for solid in self.depot.solids:
            if solid.geometry is not None:
                self.settings.add_geometry(solid.geometry)

    def _create_patches(self, block_list: BlockList) -> tuple[PatchList, FaceList]:
        patch_list = PatchList()
        face_list = FaceList()

        for i, block in enumerate(block_list.blocks):
            if not block.visible:
                continue

            patch_list.add(block.vertices, self._operations[i])
            face_list.add(block.vertices, self._operations[i])

        for name, settings in self.settings.patch_settings.items():
            patch_list.modify(name, settings[0], settings[1:])

        # set slave patches
        for pair in self.settings.merged_patches:
            patch_list.merge(pair[0], pair[1])

        return patch_list, face_list

    def _update_neighbours(self, block_list: BlockList) -> None:
        block_list.update_neighbours(self._points)

    def assemble(self) -> AssembledDump:
        # Create reused/indexes vertices from operations' points
        vertex_list = VertexList([Vertex(pos, i) for i, pos in enumerate(self._points.unique_points)])
        # Create blocks from vertices; when there's a slave patch specified in an operation,
        # duplicate vertices for that patch
        block_list = self._create_blocks(vertex_list)
        # extract edges from operations and attach them to blocks
        edge_list = self._create_edges(block_list)

        # extract auto-generated geometry specs from shapes (like Sphere etc.)
        self._add_geometry()
        # update blocks' neighbours
        self._update_neighbours(block_list)
        # scrape patch and projection info from operations
        patch_list, face_list = self._create_patches(block_list)

        return AssembledDump(vertex_list, block_list, edge_list, face_list, patch_list)
