from classy_blocks.items.block import Block
from classy_blocks.items.edges.edge import Edge
from classy_blocks.items.patch import Patch, Side
from classy_blocks.items.vertex import Vertex
from classy_blocks.lists.face_list import ProjectedFace
from classy_blocks.util.constants import vector_format


def indent(text: str, levels: int) -> str:
    """Indents 'text' by 'levels' tab characters"""
    return "\t" * levels + text + "\n"


def format_vertex(vertex: Vertex) -> str:
    """Returns a string representation to be written to blockMeshDict"""
    point = vector_format(vertex.position)
    comment = f"// {vertex.index}"

    if len(vertex.projected_to) > 0:
        return f"project {point} ({' '.join(vertex.projected_to)}) {comment}"

    return f"{point} {comment}"


def format_block(block: Block) -> str:
    if all(axis.is_simple for axis in block.axes):
        fmt_grading = (
            "simpleGrading ( "
            + block.axes[0].wires.format_single()
            + " "
            + block.axes[1].wires.format_single()
            + " "
            + block.axes[2].wires.format_single()
            + " )"
        )
    else:
        fmt_grading = (
            "edgeGrading ( "
            + block.axes[0].wires.format_all()
            + " "
            + block.axes[1].wires.format_all()
            + " "
            + block.axes[2].wires.format_all()
            + " )"
        )

    fmt_hidden = "" if block.visible else "// "
    fmt_vertices = "( " + " ".join(str(v.index) for v in block.vertices) + " )"
    fmt_count = "( " + " ".join([str(axis.count) for axis in block.axes]) + " )"

    fmt_comments = f"// {block.index} {block.comment}"

    return f"{fmt_hidden}hex {fmt_vertices} {block.cell_zone} {fmt_count} {fmt_grading} {fmt_comments}"


def format_edge(edge: Edge) -> str:
    return edge.description


def format_side(side: Side) -> str:
    return "(" + " ".join([str(v.index) for v in side.vertices]) + ")"


def format_patch(patch: Patch) -> str:
    # inlet
    # {
    #     type patch;
    #     faces
    #     (
    #         (0 1 2 3)
    #     );
    # }
    out = f"{patch.name}\n"
    out += indent("{", 1)
    out += indent(f"type {patch.kind};", 2)

    for option in patch.settings:
        out += indent(f"{option};", 2)

    out += indent("faces", 2)
    out += indent("(", 2)

    for side in patch.sides:
        out += indent(format_side(side), 3)

    out += indent(");", 2)
    out += indent("}", 1)

    return out


def format_face(face: ProjectedFace) -> str:
    return f"\tproject {format_side(face.side)} {face.label}\n"
