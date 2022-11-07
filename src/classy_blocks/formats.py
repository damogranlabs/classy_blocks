import numpy as np

from classy_blocks.define.block import Block
from classy_blocks.define.primitives import Vertex, Edge, WrongEdgeTypeException

from classy_blocks.util import constants

def format_vertex(vertex:Vertex):
    s = constants.vector_format(vertex.point)

    if vertex.mesh_index is not None:
        s += " // {}".format(vertex.mesh_index)

    return s

def format_edge(edge:Edge):
    if edge.type == "line":
        point_list = None
    elif edge.type == "project":
        point_list =  f"({edge.points})"
    elif edge.type == "arc":
        point_list = constants.vector_format(edge.points)
    elif edge.type == "spline":
        point_list = "(" + " ".join([constants.vector_format(p) for p in edge.points]) + ")"
    else:
        raise WrongEdgeTypeException(edge.type)

    return f"{edge.type} {edge.vertex_1.mesh_index} {edge.vertex_2.mesh_index} {point_list}"

def format_block(block:Block, index:int, grading:list) -> str:
    """outputs block's definition for blockMeshDict file"""
    # hex definition
    output = "hex "
    # vertices
    output += " ( " + " ".join(str(v.mesh_index) for v in block.vertices) + " ) "

    # cellZone
    output += block.cell_zone
    # number of cells
    output += f" ({grading[0].count} {grading[1].count} {grading[2].count}) "
    # grading
    output += f" simpleGrading ({grading[0]} {grading[1]} {grading[2]})"

    # add a comment with block index
    output += f" // {index} {block.description}"

    return output

def format_face(block:Block, side: int) -> str:
    indexes = block.face_map[side]
    vertices = np.take(block.vertices, indexes)

    return "({} {} {} {})".format(
        vertices[0].mesh_index, vertices[1].mesh_index, vertices[2].mesh_index, vertices[3].mesh_index
    )
