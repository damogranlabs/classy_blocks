import abc
import copy

from typing import List, Optional, Dict

from classy_blocks.types import AxisType, NPPointType, PointType, VectorType
from classy_blocks.base.transformable import TransformableBase

from classy_blocks.construct.edges import EdgeData
from classy_blocks.construct.flat.face import Face
from classy_blocks.grading.chop import Chop

class Operation(TransformableBase, abc.ABC):
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
        self.chops[axis].append(Chop(**kwargs))

    # def set_patch(self, sides: Union[str, List[str]], patch_name: str) -> None:
    #     """bottom: bottom face
    #     top: top face

    #     front: along first edge of a face
    #     back: opposite front

    #     right: along second edge of a face
    #     left: opposite right"""
    #     self.block.set_patch(sides, patch_name)

    def copy(self) -> 'Operation':
        """Returns a copy of this Operation"""
        return copy.deepcopy(self)

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
