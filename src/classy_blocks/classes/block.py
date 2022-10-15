"""Contains all data to place a block into mesh."""
from typing import List, Literal, NoReturn, Union, Tuple

import numpy as np

from classy_blocks.util import functions as f
from classy_blocks.classes.flat.face import Face
from classy_blocks.classes.primitives import Vertex, Edge
from classy_blocks.classes.grading import Grading


class Block:
    """a direct representation of a blockMesh block;
    contains all necessary data to create it."""

    # a more intuitive and quicker way to set patches,
    # according to this sketch: https://www.openfoam.com/documentation/user-guide/blockMesh.php
    # the same for all blocks
    face_map = {
        "bottom": (0, 1, 2, 3),
        "top": (4, 5, 6, 7),
        "left": (4, 0, 3, 7),
        "right": (5, 1, 2, 6),
        "front": (4, 5, 1, 0),
        "back": (7, 6, 2, 3),
    }

    # pairs of vertices (index in block.vertices) along axes
    axis_pair_indexes = (
        ((0, 1), (3, 2), (4, 5), (7, 6)),  # x
        ((0, 3), (1, 2), (5, 6), (4, 7)),  # y
        ((0, 4), (1, 5), (2, 6), (3, 7)),  # z
    )

    def __init__(self, vertices: List[Vertex], edges: List[Edge]):
        # a list of 8 Vertex and Edge objects for each corner/edge of the block
        self.vertices: List[Vertex] = vertices
        self.edges: List[Edge] = edges
        self.faces: List[List[str]] = []  # a list of projected faces;
        # [['bottom', 'terrain'], ['right', 'building'], ['back', 'building'],]

        # block grading;
        # when adding blocks, store chop() parameters;
        # use them in mesh.prepare_data()
        self.chops = [[], [], []]
        self.grading = [Grading(), Grading(), Grading()]

        # cellZone to which the block belongs to
        self.cell_zone = ""

        # written as a comment after block definition
        # (visible in blockMeshDict, useful for debugging)
        self.description = ""

        # patches: an example
        # self.patches = {
        #     'volute_rotating': ['left', 'top' ],
        #     'volute_walls': ['bottom'],
        # }
        self.patches = {}

        # set in Mesh.prepare_data()
        self.mesh_index = None

        # a list of blocks that share an edge with this block;
        # will be assigned by Mesh().prepare_data()
        self.neighbours: set["Block"] = set()

    ###
    ### Information
    ###
    @property
    def is_grading_defined(self) -> bool:
        """Returns True if grading is defined in all dimensions"""
        return all(g.is_defined for g in self.grading)

    def get_face(self, side: str, internal: bool = False) -> List[int]:
        """Returns vertex indexes for a given face.

        If internal=True, it returns block-based internal indexes (0...7);
        if internal=False, it returns indexes of vertices in Mesh object so it
        can only be called after mesh.prepare_data()"""
        indexes = self.face_map[side]
        if internal:
            return indexes

        vertices = np.take(self.vertices, indexes)

        return [v.mesh_index for v in vertices]

    def get_faces(self, patch: str, internal: bool = False) -> List[Face]:
        """Returns faces in this block that belong to a given patch"""
        if patch not in self.patches:
            return []

        sides = self.patches[patch]
        return [self.get_face(s, internal=internal) for s in sides]

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

        edge_lengths = [block_size(pair[0], pair[1]) for pair in self.axis_pair_indexes[axis]]

        if take == "avg":
            return sum(edge_lengths) / len(edge_lengths)

        if take == "min":
            return min(edge_lengths)

        if take == "max":
            return max(edge_lengths)

        raise ValueError(f"Unknown sizing specification: {take}. Available: min, max, avg")

    def get_axis_vertex_pairs(self, axis: int) -> List[int]:
        """returns 4 pairs of Vertex.mesh_indexes along given axis;
        can only be called after Mesh.prepare_data()"""
        pairs = []

        for pair in self.axis_pair_indexes[axis]:
            pair = [self.vertices[pair[0]].mesh_index, self.vertices[pair[1]].mesh_index]

            if pair[0] == pair[1]:
                # omit vertices in the same spot; there is no edge anyway
                # (wedges and pyramids)
                continue

            if pair in pairs:
                # also omit duplicates
                continue

            pairs.append(pair)

        return pairs

    def get_axis_from_pair(self, pair: List[int]) -> Tuple[int, bool]:
        """returns axis index and orientation from a given pair of vertices;
        orientation is True if blocks are aligned or false when inverted.

        This can only be called after Mesh.prepare_data()"""
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
    def set_patch(self, sides: Union[str, List[str]], patch_name: str) -> NoReturn:
        """assign one or more block faces (self.face_map)
        to a chosen patch name"""
        # see patches: an example in __init__()

        if isinstance(sides, str):
            sides = [sides]

        if patch_name not in self.patches:
            self.patches[patch_name] = []

        self.patches[patch_name] += sides

    def chop(self, axis: int, **kwargs: float) -> NoReturn:
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

    def grade(self) -> NoReturn:
        """Sets block size and grading; not to be used manually!"""
        for i in range(3):
            grading = self.grading[i]
            params = self.chops[i]

            if len(params) < 1:
                continue

            block_size = self.get_size(i, take=params[0].pop("take", "avg"))
            grading.set_block_size(block_size)

            for p in params:
                grading.add_division(**p)

        self.chops = [[], [], []]

    def project_edge(self, index_1: int, index_2: int, geometry: str) -> NoReturn:
        """Project a block edge between index_1 and index_2 to geometry (specified in Mesh)
        Indexes refer to refer to internal block numbering (0...7)."""
        # index_N are vertices relative to block (0...7)
        if self.find_edge(index_1, index_2):
            return

        self.edges.append(Edge(index_1, index_2, geometry))

    def project_face(
        self, side: Literal["top", "bottom", "left", "right", "front", "back"], geometry: str, edges: bool = False
    ) -> NoReturn:
        """Assign one or more block faces (self.face_map)
        to be projected to a geometry (defined in Mesh)"""
        assert side in self.face_map

        self.faces.append([side, geometry])

        if edges:
            vertices = self.face_map[side]
            for i in range(4):
                self.project_edge(vertices[i], vertices[(i + 1) % 4], geometry)

    ###
    ### Output/formatting
    ###
    def format_face(self, side: int) -> str:
        """Returns a string to be inserted into blockMesh"""
        indexes = self.face_map[side]
        vertices = np.take(self.vertices, indexes)

        return "({} {} {} {})".format(
            vertices[0].mesh_index, vertices[1].mesh_index, vertices[2].mesh_index, vertices[3].mesh_index
        )

    @property
    def n_cells(self) -> List[int]:
        """Returns number of cells for each axis"""
        return [g.count for g in self.grading]

    def __repr__(self):
        """outputs block's definition for blockMeshDict file"""
        # hex definition
        output = "hex "
        # vertices
        output += " ( " + " ".join(str(v.mesh_index) for v in self.vertices) + " ) "

        # cellZone
        output += self.cell_zone
        # number of cells
        output += f" ({self.n_cells[0]} {self.n_cells[1]} {self.n_cells[2]}) "
        # grading: use 1 if not defined
        grading = [self.grading[i] if self.grading[i] is not None else 1 for i in range(3)]
        output += f" simpleGrading ({grading[0]} {grading[1]} {grading[2]})"

        # add a comment with block index
        output += f" // {self.mesh_index} {self.description}"

        return output

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
