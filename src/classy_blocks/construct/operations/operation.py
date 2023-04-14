from typing import List, Dict, Union, TypeVar, Set, Tuple

import numpy as np

from classy_blocks.types import AxisType, NPPointType, OrientType, ProjectToType

from classy_blocks.base.element import ElementBase

from classy_blocks.items.frame import Frame

from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.edges import EdgeData, Project
from classy_blocks.grading.chop import Chop

from classy_blocks.util import constants

OperationT = TypeVar("OperationT", bound="Operation")


class Operation(ElementBase):
    """A base class for all single-block operations
    (Box, Loft, Revolve, Extrude, Wedge)."""

    SIDES_MAP = ("front", "right", "back", "left")  # connects orients/indexes of side faces/edges

    def __init__(self, bottom_face: Face, top_face: Face):
        # An Operation only holds data to create the block;
        # actual object is created by Mesh when it is added.

        # Top and bottom faces contain all points:
        self.faces: Dict[OrientType, Face] = {}
        self.faces["bottom"] = bottom_face
        self.faces["top"] = top_face

        # side edges, patch names and projections
        # are held within the other four faces but
        # points and edges are duplicated; therefore
        # some care must be taken when getting information
        # from an Operation:
        #  - 8 points at corners from top and bottom faces
        #  - edges on top and bottom from corresponding faces
        #  - side edges are the first edge in each of left/right/front/back faces
        #  - projected faces are not duplicated, each face holds its own info
        # see self.set_* methods and properties
        for orient in self.SIDES_MAP:
            self.faces[orient] = Face([self.point_array[i] for i in constants.FACE_MAP[orient]])

        # instructions for cell counts and gradings
        self.chops: Dict[AxisType, List[Chop]] = {0: [], 1: [], 2: []}

        # optionally, put the block in a cell zone
        self.cell_zone = ""

    def add_side_edge(self, corner_1: int, edge_data: EdgeData) -> None:
        """Add an edge between two vertices at the same
        corner of the lower and upper face (index and index+4 or vice versa)."""
        assert corner_1 < 4, "corner_1 must be an index to a bottom Vertex (0...3)"

        # TODO: TEST
        orient = self.SIDES_MAP[corner_1]
        self.faces[orient].add_edge(0, edge_data)

    def chop(self, axis: AxisType, **kwargs) -> None:
        """Chop the operation (count/grading) in given axis:
        0: along first edge of a face
        1: along second edge of a face
        2: between faces / along operation path

        Kwargs: see arguments for Chop object"""
        self.chops[axis].append(Chop(**kwargs))

    def unchop(self, axis: AxisType) -> None:
        """Removed existing chops from an operation
        (comes handy after copying etc.)"""
        self.chops[axis] = []

    def project_corner(self, corner: int, geometry: ProjectToType) -> None:
        """Project the vertex at given corner (local index 0...7) to a single
        surface or an intersection of multiple surface. WIP according to
        https://github.com/OpenFOAM/OpenFOAM-10/blob/master/src/meshTools/searchableSurfaces/searchableSurfacesQueries/searchableSurfacesQueries.H
        """
        # bottom and top faces define operation's points
        if corner > 3:
            self.faces["top"].points[corner - 4].project(geometry)
        else:
            self.faces["bottom"].points[corner].project(geometry)

    def project_edge(self, corner_1: int, corner_2: int, geometry: Union[str, List[str]]) -> None:
        """Project an edge to a surface or an intersection of two surfaces"""

        face, corner = self.edge_map[corner_1][corner_2]
        face.edges[corner] = Project(geometry)

    def project_side(self, side: OrientType, geometry: str, edges: bool = False, points: bool = False) -> None:
        """Project given side to named geometry;

        Args:
        - side: 'bottom', 'top', 'front', 'back', 'left', 'right';
            only
            the sketch from blockMesh documentation:
            https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility
            bottom, top: faces from which the Operation was created
            front: along first edge of a face
            back: opposite front
            right: along second edge of a face
            left: opposite right
        - geometry: name of predefined geometry (add separately to Mesh object)
        - edges:if True, all edges belonging to this side will also be projected"""
        self.faces[side].project(geometry, edges, points)

    def set_patch(self, sides: Union[OrientType, List[OrientType]], name: str) -> None:
        """Assign a patch to given side of the block;

        Args:
        - side: 'bottom', 'top', 'front', 'back', 'left', 'right',
            a single value or a list of sides; names correspond to position in
            the sketch from blockMesh documentation:
            https://www.openfoam.com/documentation/user-guide/4-mesh-generation-and-conversion/4.3-mesh-generation-with-the-blockmesh-utility
            bottom, top: faces from which the Operation was created
            front: along first edge of a face
            back: opposite front
            right: along second edge of a face
            left: opposite right
        - name: the name that goes into blockMeshDict

        Use mesh.set_patch_* methods to change other properties (type and other settings)"""
        if not isinstance(sides, list):
            sides = [sides]

        for orient in sides:
            self.faces[orient].patch_name = name

    def set_cell_zone(self, cell_zone: str) -> None:
        """Assign a cellZone to this block."""
        self.cell_zone = cell_zone

    @property
    def parts(self):
        return self.faces.values()

    @property
    def point_array(self) -> NPPointType:
        """Returns 8 points from which this operation is created"""
        return np.concatenate((self.faces["bottom"].point_array, self.faces["top"].point_array))

    @property
    def center(self):
        # TODO: TEST
        return np.average(self.point_array, axis=0)

    def get_patches_at_corner(self, corner: int) -> Set[str]:
        """Returns patch names at given corner (up to 3)"""
        patches = set()

        for orient, corners in constants.FACE_MAP.items():
            if corner in corners:
                if self.faces[orient].patch_name is not None:
                    patches.add(self.faces[orient].patch_name)

        return patches

    @property
    def patch_names(self) -> Dict[OrientType, str]:
        """Returns patches names on sides where they are specified"""
        patch_names: Dict[OrientType, str] = {}

        for orient, face in self.faces.items():
            if face.patch_name is not None:
                patch_names[orient] = face.patch_name

        return patch_names

    @property
    def edge_map(self) -> Frame[Tuple[Face, int]]:
        """Edge addressing: map the two corners of edges to
        corresponding face"""
        # TODO: TEST
        edge_map = Frame[Tuple[Face, int]]()

        for i in range(4):
            # bottom face
            edge_map.add_beam(i, (i + 1) % 4, (self.faces["bottom"], i))
            # top face
            edge_map.add_beam(4 + i, 4 + (i + 1) % 4, (self.faces["top"], i))
            # sides
            orient = self.SIDES_MAP[i]
            edge_map.add_beam(i, i + 4, (self.faces[orient], i))

        return edge_map

    @property
    def edges(self) -> Frame[EdgeData]:
        """Returns a Frame with edges as its beams"""
        edges = Frame[EdgeData]()

        for i, data in enumerate(self.faces["bottom"].edges):
            if data.kind != "line":
                edges.add_beam(i, (i + 1) % 4, data)

        for i, data in enumerate(self.faces["top"].edges):
            if data.kind != "line":
                edges.add_beam(4 + i, 4 + (i + 1) % 4, data)

        for i, orient in enumerate(self.SIDES_MAP):
            data = self.faces[orient].edges[0]
            if data.kind != "line":
                edges.add_beam(i, i + 4, data)

        return edges
