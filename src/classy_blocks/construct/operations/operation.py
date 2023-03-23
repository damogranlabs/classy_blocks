from typing import List, Optional, Dict, Union, TypeVar

import numpy as np

from classy_blocks.types import AxisType, NPPointType, PointType, VectorType, OrientType
from classy_blocks.base.additive import AdditiveBase

from classy_blocks.construct.operations.projections import ProjectedEntities
from classy_blocks.construct.flat.face import Face
from classy_blocks.construct.edges import EdgeData, EdgeInfo
from classy_blocks.grading.chop import Chop

from classy_blocks.util import constants

OperationT = TypeVar('OperationT', bound='Operation')

class Operation(AdditiveBase):
    """A user-friendly way to create a Block, as a 2-point Box,
    extruded/revolved from a single Face or Lofted between two faces
    with optional side edges."""
    def __init__(self, bottom_face: Face, top_face: Face):
        # An Operation only holds data to create the block;
        # actual object is created by Mesh when it is added.
        self.bottom_face = bottom_face
        self.top_face = top_face
        self.side_edges:List[Optional[EdgeData]] = [None]*4
        self.chops:Dict[AxisType, List[Chop]] = {0: [], 1: [], 2: []}
        self.patch_names:Dict[OrientType, str] = {}

        self.projections = ProjectedEntities()

        self.cell_zone = "" # set with self.cell_zone

    def add_side_edge(self, corner_1:int, edge_data:EdgeData) -> None:
        """Add an edge between two vertices at the same
        corner of the lower and upper face (index and index+4 or vice versa)."""
        assert corner_1 < 4, "corner_1 must be an index to a bottom Vertex (0...3)"
        self.side_edges[corner_1] = edge_data

    def chop(self, axis:AxisType, **kwargs) -> None:
        """Chop the operation (count/grading) in given axis:
        0: along first edge of a face
        1: along second edge of a face
        2: between faces / along operation path

        Kwargs: see arguments for Chop object"""
        # TODO: TEST
        self.chops[axis].append(Chop(**kwargs))

    def unchop(self, axis:AxisType) -> None:
        """Removed existing chops from an operation
        (comes handy after copying etc.)"""
        self.chops[axis] = []

    def set_patch(self, sides: Union[OrientType, List[OrientType]], name:str) -> None:
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
            self.patch_names[orient] = name

    def project_side(self, side:OrientType, geometry:str, edges:bool=False) -> None:
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
        # TODO: TEST
        self.projections.add_side(side, geometry)

        if edges:
            for i, corner in enumerate(constants.FACE_MAP[side]):
                self.project_edge(
                    corner,
                    constants.FACE_MAP[side][(i+1) % 4], geometry
                )

    def project_edge(self, corner_1:int, corner_2:int, geometry:Union[str, List[str]]) -> None:
        """Project an edge to a surface or an intersection of two surfaces"""
        self.projections.add_edge(corner_1, corner_2, geometry)

    def project_corner(self, corner:int, geometry:Union[str, List[str]]) -> None:
        """Project the vertex at given corner (local index 0...7) to a single
        surface or an intersection of multiple surface. WIP according to
        https://github.com/OpenFOAM/OpenFOAM-10/blob/master/src/meshTools/searchableSurfaces/searchableSurfacesQueries/searchableSurfacesQueries.H"""
        self.projections.add_vertex(corner, geometry)

    def translate(self, displacement: VectorType) -> 'Operation':
        """returns a translated copy of this Operation"""
        self.bottom_face.translate(displacement)
        self.top_face.translate(displacement)

        for edge in self.side_edges:
            if edge is not None:
                edge.translate(displacement)

        return self

    def rotate(self, angle: float, axis: VectorType, origin: Optional[PointType] = None) -> 'Operation':
        """Rotate this Operation by a specified angle around given axis;
        if origin is not specified, center of the Operation is taken"""
        if origin is None:
            origin = self.center

        self.bottom_face.rotate(angle, axis, origin)
        self.top_face.rotate(angle, axis, origin)

        for edge in self.side_edges:
            if edge is not None:
                edge.rotate(angle, axis, origin)

        return self

    def scale(self, ratio: float, origin: Optional[PointType] = None) -> 'Operation':
        """Scale this Operation by a given ratio.
        If origin is not specified, center of Operation is taken"""
        if origin is None:
            origin = self.center

        self.bottom_face.scale(ratio, origin)
        self.top_face.scale(ratio, origin)

        for edge in self.side_edges:
            if edge is not None:
                edge.scale(ratio, origin)

        return self

    def set_cell_zone(self, cell_zone: str) -> None:
        """Assign a cellZone to this block."""
        self.cell_zone = cell_zone

    @property
    def center(self) -> NPPointType:
        """Center of this object"""
        return (self.bottom_face.center + self.top_face.center)/2

    @property
    def operations(self:OperationT) -> List[OperationT]:
        return [self]

    @property
    def points(self) -> NPPointType:
        """Returns 8 points from which this operation is created"""
        return np.concatenate((self.bottom_face.points, self.top_face.points))

    @property
    def edges(self) -> List[EdgeInfo]:
        """Returns assembled EdgeInfo from faces, sides and projections"""
        raise NotImplemented

    @property
    def faces(self) -> Dict[OrientType, Face]:
        """Create a new Face from points on given 'side' of this Operation;
        does not copy edges (they don't need to be copied)"""
        # TODO: TEST

        return {
            side:Face([self.points[i] for i in corners])
            for side, corners in constants.FACE_MAP.items()
        }

    def get_patch_from_corner(self, corner:int) -> Optional[str]:
        """Returns patch name at given corner or None if
        no patch has been defined"""
        for side, patch in self.patch_names.items():
            if corner in constants.FACE_MAP[side]:
                return patch
        
        return None
