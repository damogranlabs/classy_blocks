"""Contains all data that user can specify for any kind of Edge"""
import dataclasses
from typing import Union, List

from classy_blocks.types import EdgeKindType

from classy_blocks.data.point import Point
from classy_blocks.base.transformable import TransformableBase

@dataclasses.dataclass
class EdgeData(TransformableBase):
    """User-provided data for an edge
    Constructor args:
    :param corner_1: block-local index to block.points
    :param corner_2: block-local index to block.points
    :param kind: edge type
    :param args: optional additional arguments for given edge
    
    The 'args' parameter should contain data based on edge type:
     - line: nothing, empty list
     - arc: a single point
     - angle: [angle, axis vector]
     - origin: [origin point, optional flatness]
     - spline: [a list of points]
     - polyLine: [a list of points]
     - project: [names of geometries to project to]
    """
    corner_1:int
    corner_2:int
    kind:EdgeKindType
    args:List[Union[Point, float, str]]

    def __post_init__(self):
        # Create Point objects where needed
        # TODO: factory or something?
        match self.kind:
            case 'arc':
                self.args[0] = Point(self.args[0])
            case 'angle':
                self.args[1] = Point(self.args[1])
            case 'origin':
                self.args[0] = Point(self.args[0])
            case ('spline', 'polyLine'):
                self.args = [Point(p) for p in self.args]

    @property
    def points(self):
        return [p for p in self.args if isinstance(p, Point)]

