import abc

from typing import Type, ClassVar

from classy_blocks.process.items.vertex import Vertex
from classy_blocks.define.curve import Curve

import dataclasses

@dataclasses.dataclass
class Edge(abc.ABC):
    vertex_1:Vertex
    vertex_2:Vertex
    curve:Curve

    kind: ClassVar[str] = ''

    @abc.abstractmethod
    def output(self) -> str:
        """String for blockMeshDict"""
        return "TODO"

class ArcEdge(Edge):
    pass

#     def __init__(self, definition:EdgeDefinition):
#         super().__init__(definition)

#         self.point = definition.data

#     @property
#     def is_valid(self):
#         """Checks if this arc is a line;"""
#         if not super().is_valid:
#             return False

#         # in case vertex1, vertex2 and point in between
#         # are collinear, blockMesh will find an arc with
#         # infinite radius and crash.
#         # so, check for collinearity; if the three points
#         # are actually collinear, this edge is redundant and can be
#         # silently dropped
#         OA = self.vertex_1.point
#         OB = self.vertex_2.point
#         OC = self.point

#         # if point C is on the same line as A and B:
#         # OC = OA + k*(OB-OA)
#         AB = OB - OA
#         AC = OC - OA

#         k = f.norm(AC) / f.norm(AB)
#         d = f.norm((OA + AC) - (OA + k * AB))

#         return d > constants.tol

#     @property
#     def length(self) -> float:
#         """Returns an approximate length of this Edge"""
#         assert self.vertex_1 is not None and self.vertex_2 is not None, "Unprepared mesh"

#         return f.arc_length_3point(self.vertex_1.point, self.point, self.vertex_2.point)


#     def translate(self, displacement:VectorType):
#         """Move all points in the edge (but not start and end) 
#         by a displacement vector."""
#         displacement = np.asarray(displacement)

#         new_def = self.definition.copy()
#         new_def.data = self.point + displacement

#         return self.__class__(new_def)

#     def rotate(self, angle:float, axis:VectorType, origin:Optional[PointType]=None):
#         """ Rotates all points in this edge (except start and end Vertex) around an
#         arbitrary axis and origin (be careful with projected edges, geometry isn't rotated!)"""
#         if origin is None:
#             origin = [0, 0, 0]

#         new_def = self.definition.copy()
#         new_def.data = f.arbitrary_rotation(self.point, axis, angle, origin)

#         return self.__class__(new_def)

#     def scale(self, ratio:float, origin:PointType):
#         """Scales the edge points around given origin"""
#         new_def = self.definition.copy()
#         new_def.data = Vertex.scale_point(self.point, ratio, origin)

#         return self.__class__(new_def)

#     def output(self) -> str:
#         return f"arc {self.vertex_1.index} {self.vertex_2.index} ({constants.vector_format(self.point)})"


class OriginEdge(ArcEdge):
    pass

#     def __init__(self, definition:EdgeDefinition):
#         assert definition.point_1 is not None
#         assert definition.point_2 is not None
#         assert definition.origin is not None

#         definition.data = arc_from_origin(
#             definition.point_1,
#             definition.point_2,
#             definition.origin,
#             r_multiplier=definition.flatness)

#         super().__init__(definition)

#     def translate(self, displacement:VectorType):
#         """Move all points in the edge (but not start and end)
#         by a displacement vector."""
#         displacement = np.asarray(displacement)

#         new_def = self.definition.copy()
#         new_def.origin += displacement

#         return self.__class__(new_def)

#     def rotate(self, angle:float, axis:VectorType, origin:Optional[PointType]=None):
#         """ Rotates all points in this edge (except start and end Vertex) around an
#         arbitrary axis and origin (be careful with projected edges, geometry isn't rotated!)"""
#         if origin is None:
#             origin = [0, 0, 0]

#         new_def = self.definition.copy()
#         new_def.origin = f.arbitrary_rotation(self.definition.origin, axis, angle, origin)

#         return self.__class__(new_def)

#     def scale(self, ratio:float, origin):
#         """Scales the edge points around given origin"""
#         new_def = self.definition.copy()
#         new_def.data = Vertex.scale_point(self.point, ratio, origin)

#         return self.__class__(new_def)

#     def output(self) -> str:
#         return f"arc {self.vertex_1.index} {self.vertex_2.index} ({constants.vector_format(self.point)})"

class AngleEdge(ArcEdge):
    pass

#     def __init__(self, definition:EdgeDefinition):
#         assert definition.point_1 is not None
#         assert definition.point_2 is not None
#         assert definition.angle is not None
#         assert definition.axis is not None

#         definition.data = arc_from_theta(
#             definition.point_1,
#             definition.point_2,
#             definition.angle,
#         definition.axis)

#         super().__init__(definition)

#     def translate(self, displacement:VectorType):
#         """Move all points in the edge (but not start and end)
#         by a displacement vector."""
#         displacement = np.asarray(displacement)

#         new_def = self.definition.copy()
#         new_def.origin += displacement

#         return self.__class__(new_def)

#     def rotate(self, angle:float, axis:VectorType, origin:Optional[PointType]=None):
#         """ Rotates all points in this edge (except start and end Vertex) around an
#         arbitrary axis and origin (be careful with projected edges, geometry isn't rotated!)"""
#         if origin is None:
#             origin = [0, 0, 0]

#         new_def = self.definition.copy()
#         new_def.origin = f.arbitrary_rotation(self.definition.origin, axis, angle, origin)

#         return self.__class__(new_def)

#     def scale(self, ratio:float, origin):
#         """Scales the edge points around given origin"""
#         new_def = self.definition.copy()
#         new_def.data = Vertex.scale_point(self.point, ratio, origin)

#         return self.__class__(new_def)

#     def output(self) -> str:
#         return f"arc {self.vertex_1.index} {self.vertex_2.index} ({constants.vector_format(self.point)})"



class SplineEdge(Edge):
    pass

#     def __init__(self, definition:EdgeDefinition):
#         super().__init__(definition)

#         self.points = definition.data
#         # assert len(np.shape(self.points)) == 2, f"Supply a list of points for a {self.kind} edge"

#     @property
#     def length(self) -> float:
#         """Calculates the length of this edge by taking
#         linear distance between points"""
#         edge_points = np.concatenate((
#             [self.vertex_1.point],
#             self.points,
#             [self.vertex_2.point]), axis=0)

#         l = 0.0

#         for i in range(len(edge_points) - 1):
#             l += f.norm(edge_points[i + 1] - edge_points[i])

#         return l

#     def translate(self, displacement:VectorType):
#         """Move all points in the edge (but not start and end) 
#         by a displacement vector."""
#         displacement = np.asarray(displacement)
        
#         new_points = [e + displacement for e in self.points]
        
#         #return self.__class__(self.index_1, self.index_2, new_points, self.kind)

#     def rotate(self, angle:List[float], axis:List[float], origin:List[float]=None):
#         """ Rotates all points in this edge (except start and end Vertex) around an
#         arbitrary axis and origin (be careful with projected edges, geometry isn't rotated!)"""
#         if origin is None:
#             origin = [0, 0, 0]

#         points = [f.arbitrary_rotation(p, axis, angle, origin) for p in self.points]
        
#         #return self.__class__(self.index_1, self.index_2, points, kind=self.kind)

#     def scale(self, ratio:float, origin):
#         """Scales the edge points around given origin"""
#         points = [Vertex.scale_point(p, ratio, origin) for p in self.points]
        
#         #return self.__class__(self.index_1, self.index_2, points, kind=self.kind)

#     def output(self) -> str:
#         return f"arc {vertex_1.mesh_index} {vertex_2.mesh_index} ({constants.vector_format(self.point)})"


class PolyLineEdge(SplineEdge):
    pass

class ProjectedEdge(Edge):
    pass

#     def __init__(self, definition:EdgeDefinition):
#         super().__init__()

#         self.geometry = definition.data

#     def get_length(self) -> float:
#         """Returns an approximate length of this Edge"""
#         # since we don't know the geometry this will be projected to,
#         # the best guess is to just get the distance between endpoints
#         return f.norm(self.vertex_1.point - self.vertex_2.point)

class EdgeFactory:
    def __init__(self):
        self.kinds = {}

    def register_kind(self, creator:Type[Edge]) -> None:
        """Introduces a new edge kind to this factory"""
        self.kinds[creator.curve.kind] = creator

    def create(self, vertex_1:Vertex, vertex_2:Vertex, curve:Curve):
        kind = self.kinds[curve.kind]
        return kind(vertex_1, vertex_2, curve)

factory = EdgeFactory()
factory.register_kind(ArcEdge)
factory.register_kind(OriginEdge)
factory.register_kind(AngleEdge)
factory.register_kind(SplineEdge)
factory.register_kind(PolyLineEdge)
factory.register_kind(ProjectedEdge)
