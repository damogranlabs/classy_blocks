import abc

from classy_blocks.base.exceptions import MeshNotAssembledError
from classy_blocks.items.block import Block
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.patch import Patch
from classy_blocks.items.vertex import Vertex
from classy_blocks.lists.block_list import BlockList
from classy_blocks.lists.edge_list import EdgeList
from classy_blocks.lists.face_list import FaceList
from classy_blocks.lists.patch_list import PatchList
from classy_blocks.lists.vertex_list import VertexList


class DumpBase(abc.ABC):
    @property
    @abc.abstractmethod
    def is_assembled(self) -> bool:
        pass

    @property
    @abc.abstractmethod
    def vertices(self) -> list[Vertex]:
        pass

    @property
    @abc.abstractmethod
    def patches(self) -> list[Patch]:
        pass

    @property
    @abc.abstractmethod
    def blocks(self) -> list[Block]:
        pass

    @property
    @abc.abstractmethod
    def edges(self) -> list[Edge]:
        pass

    @abc.abstractmethod
    def finalize(self) -> None:
        pass


class EmptyDump(DumpBase):
    @property
    def is_assembled(self):
        return False

    @property
    def vertices(self):
        raise MeshNotAssembledError("The mesh is not assembled")

    @property
    def patches(self):
        raise MeshNotAssembledError("The Mesh is not assembled")

    @property
    def blocks(self):
        raise MeshNotAssembledError("The Mesh is not assembled")

    @property
    def edges(self):
        raise MeshNotAssembledError("The Mesh is not assembled")

    def finalize(self):
        raise MeshNotAssembledError("The Mesh is not assembled")


class AssembledDump(DumpBase):
    def __init__(
        self,
        vertex_list: VertexList,
        block_list: BlockList,
        edge_list: EdgeList,
        face_list: FaceList,
        patch_list: PatchList,
    ):
        self.vertex_list = vertex_list
        self.block_list = block_list
        self.edge_list = edge_list
        self.face_list = face_list
        self.patch_list = patch_list

    @property
    def is_assembled(self):
        """Returns True if assemble() has been executed on this mesh"""
        return True

    @property
    def vertices(self):
        return self.vertex_list.vertices

    @property
    def patches(self):
        return list(self.patch_list.patches.values())

    @property
    def blocks(self):
        return self.block_list.blocks

    @property
    def edges(self):
        return list(self.edge_list.edges.values())

    def finalize(self):
        self.block_list.assemble()
