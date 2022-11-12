"""Contains all data to place a block into mesh."""
from typing import List, Literal, Union, Tuple, Optional, Dict

import warnings

from classy_blocks.util import functions as f
from classy_blocks.util import constants as c
from classy_blocks.construct.flat.face import Face
from classy_blocks.define.primitives import Vertex, Edge

class Side:
    """Data about one of block's sides"""
    def __init__(self, orient:Literal['left', 'right', 'front', 'back', 'top', 'bottom']):
        self.orient = orient

        # whether this block side belongs to a patch
        self.patch:Optional[str] = None
        # project to a named searchable surface?
        self.project:Optional[str] = None

class Block:
    """a direct representation of a blockMesh block;
    contains all necessary data to create it."""
    def __init__(self, vertices: List[Vertex], edges: List[Edge]):
        # a list of 8 Vertex and Edge objects for each corner/edge of the block
        self.vertices: List[Vertex] = vertices
        self.edges: List[Edge] = edges

        # generate Side objects:
        self.sides:Dict[Side] = {o:Side(o) for o in c.FACE_MAP}

        # block grading;
        # when adding blocks, store chop() parameters;
        # use them in mesh.write()
        self.chops = [[], [], []]

        # cellZone to which the block belongs to
        self.cell_zone = ""

        # written as a comment after block definition
        # (visible in blockMeshDict, useful for debugging)
        self.description = ""

        # index of this block in mesh.blocks;
        # does not really belong to this object but
        # debugging without it is quite impossible
        self.index = None

    ###
    ### Information
    ###
    def get_side_vertices(self, orient:str) -> List[Vertex]:
        """Returns Vertices that define the given side"""
        return [self.vertices[i] for i in self.get_side_indexes(orient, local=True)]

    def get_side_indexes(self, orient:str, local:bool=False) -> List[int]:
        """Returns block-local indexes of vertices that define this side"""
        if local:
            return c.FACE_MAP[orient]

        side_vertices = [self.vertices[i] for i in c.FACE_MAP[orient]]
        return [v.mesh_index for v in side_vertices]

    def get_patch_sides(self, patch: str) -> List[str]:
        """Returns sides in this block that belong to a given patch"""
        if patch not in self.patches:
            return []

        orients = []

        for orient in c.FACE_MAP:
            if self.sides[orient].patch == patch:
                orients.append(orient)

        return orients

    def find_edge(self, index_1: int, index_2: int) -> Edge:
        """Returns edges between given vertex indexes;
        the indexes in parameters refer to internal block numbering"""
        for e in self.edges:
            if {e.block_index_1, e.block_index_2} == {index_1, index_2}:
                return e

        return None

    def get_size(self, axis: int, take: Literal["min", "max", "avg"] = "avg") -> float:
        """Returns block dimensions in given axis"""
        # if an edge is defined, use the edge.get_length(),
        # otherwise simply distance between two points
        def vertex_distance(index_1, index_2):
            return f.norm(self.vertices[index_1].point - self.vertices[index_2].point)

        def block_size(index_1, index_2):
            edge = self.find_edge(index_1, index_2)
            if edge:
                return edge.get_length()

            return vertex_distance(index_1, index_2)

        edge_lengths = [block_size(pair[0], pair[1]) for pair in c.AXIS_PAIRS[axis]]

        if take == "avg":
            return sum(edge_lengths) / len(edge_lengths)

        if take == "min":
            return min(edge_lengths)

        if take == "max":
            return max(edge_lengths)

        raise ValueError(f"Unknown sizing specification: {take}. Available: min, max, avg")

    def get_axis_vertex_pairs(self, axis: int) -> List[List[Vertex]]:
        """Returns 4 pairs of Vertex objects along given axis"""
        pairs = []

        for pair in c.AXIS_PAIRS[axis]:
            pair = [self.vertices[pair[0]], self.vertices[pair[1]]]

            if pair[0] == pair[1]:
                # omit vertices in the same spot; there is no edge anyway
                # (prisms/wedges/pyramids)
                continue

            if pair in pairs:
                # also omit duplicates
                continue

            pairs.append(pair)

        return pairs

    def get_axis_from_pair(self, pair: List[Vertex]) -> Tuple[int, bool]:
        """returns axis index and orientation from a given pair of vertices;
        orientation is True if blocks are aligned or false when inverted.

        This can only be called after Mesh.write()"""
        riap = [pair[1], pair[0]]

        for i in range(3):
            pairs = self.get_axis_vertex_pairs(i)

            if pair in pairs:
                return i, True

            if riap in pairs:
                return i, False

        return None, None

    ###
    ### Manipulation
    ###
    def set_patch(self, orients: Union[str, List[str]], patch_name: str) -> None:
        """assign one or more block faces (constants.FACE_MAP)
        to a chosen patch name"""
        # see patches: an example in __init__()

        if isinstance(orients, str):
            orients = [orients]

        for orient in orients:
            if self.sides[orient].patch is not None:
                warnings.warn(f"Replacing patch {self.sides[orient].patch} with {patch_name}")

            self.sides[orient].patch = patch_name

    @property
    def patches(self) -> Dict[str, List[str]]:
        """Returns a dict of patches, for example:

        patches = {
             'volute_rotating': ['left', 'top' ],
             'volute_walls': ['bottom'],
        }"""
        # TODO: set type and other patch properties in mesh.boundary
        # TODO: test
        pdict = {}

        for orient, side in self.sides.items():
            if side.patch is not None:
                if side.patch not in pdict:
                    pdict[side.patch] = []

                pdict[side.patch].append(orient)

        return pdict


    def chop(self, axis: int, **kwargs: float) -> None:
        """Set block's cell count/size and grading for a given direction/axis.
        Exactly two of the following keyword arguments must be provided:

        :Keyword Arguments:
        * *start_size:
            size of the first cell (last if invert==True)
        * *end_size:
            size of the last cell
        * *c2c_expansion:
            cell-to-cell expansion ratio
        * *count:
            number of cells
        * *total_expansion:
            ratio between first and last cell size

        :Optional keyword arguments:
        * *invert:
            reverses grading if True
        * *take:
            must be 'min', 'max', or 'avg'; takes minimum or maximum edge
            length for block size calculation, or average of all edges in given direction.
            With multigrading only the first 'take' argument is used, others are copied.
        * *length_ratio:
            in case the block is graded using multiple gradings, specify
            length of current division; see
            https://cfd.direct/openfoam/user-guide/v9-blockMesh/#multi-grading
        """
        # when this function is called, block edges are not known and so
        # block size can't be calculated; at this point only store parameters
        # and call the actual Grading.chop() function later with these params
        self.chops[axis].append(kwargs)

    def project_edge(self, index_1: int, index_2: int, geometry: str) -> None:
        """Project a block edge between index_1 and index_2 to geometry (specified in Mesh)
        Indexes refer to refer to internal block numbering (0...7)."""
        # index_N are vertices relative to block (0...7)
        if self.find_edge(index_1, index_2):
            return

        self.edges.append(Edge(index_1, index_2, geometry))

    def project_face(
        self, orient: Literal["top", "bottom", "left", "right", "front", "back"], geometry: str, edges: bool = False
    ) -> None:
        """Assign one or more block faces (self.face_map)
        to be projected to a geometry (defined in Mesh)"""
        assert orient in c.FACE_MAP

        self.sides[orient].project = geometry

        if edges:
            vertices = c.FACE_MAP[orient]
            for i in range(4):
                self.project_edge(vertices[i], vertices[(i + 1) % 4], geometry)

    ###
    ### class methods
    ###
    @classmethod
    def create_from_points(cls, points, edges=None) -> "Block":
        """create a block from a raw list of 8 points;
        edges are optional; edge's 2 vertex indexes refer to
        block.vertices list (0 - 7)"""
        if edges is None:
            edges = []

        block = cls([Vertex(p) for p in points], edges)

        return block
