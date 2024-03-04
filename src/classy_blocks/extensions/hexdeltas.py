"""functions specifically used by hexmesh for delta calculations"""

import numpy as np
import scipy.linalg
from scipy.spatial import KDTree
from stl import Mesh as stlMesh

from classy_blocks.extensions import hexprojections as hexp
from classy_blocks.extensions import hextools as hext
from classy_blocks.mesh import Mesh
from classy_blocks.types import NPPointListType, NPPointType, OrientType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


def point_delta(
    wire_delta: list[NPPointType],
    face_delta_edge: dict[OrientType, NPPointType],
    face_delta_proj: dict[OrientType, NPPointType],
    face_proj_bool: dict[OrientType, bool],
    x_ratio: float = 0,
    y_ratio: float = 0,
    z_ratio: float = 0,
) -> NPPointType | None:
    """Calculate the point delta from the projected face deltas"""

    # calculate the delta corrections on each face from the edges

    face_delta_edge["left"] = 0.5 * (
        wire_delta[4] * (1.0 - z_ratio)
        + wire_delta[7] * (z_ratio)
        + wire_delta[8] * (1.0 - y_ratio)
        + wire_delta[11] * (y_ratio)
    )

    face_delta_edge["right"] = 0.5 * (
        wire_delta[5] * (1.0 - z_ratio)
        + wire_delta[6] * (z_ratio)
        + wire_delta[9] * (1.0 - y_ratio)
        + wire_delta[10] * (y_ratio)
    )

    face_delta_edge["front"] = 0.5 * (
        wire_delta[0] * (1.0 - z_ratio)
        + wire_delta[3] * (z_ratio)
        + wire_delta[8] * (1.0 - x_ratio)
        + wire_delta[9] * (x_ratio)
    )

    face_delta_edge["back"] = 0.5 * (
        wire_delta[1] * (1.0 - z_ratio)
        + wire_delta[2] * (z_ratio)
        + wire_delta[11] * (1.0 - x_ratio)
        + wire_delta[10] * (x_ratio)
    )

    face_delta_edge["bottom"] = 0.5 * (
        wire_delta[0] * (1.0 - y_ratio)
        + wire_delta[1] * (y_ratio)
        + wire_delta[4] * (1.0 - x_ratio)
        + wire_delta[5] * (x_ratio)
    )

    face_delta_edge["top"] = 0.5 * (
        wire_delta[3] * (1.0 - y_ratio)
        + wire_delta[2] * (y_ratio)
        + wire_delta[7] * (1.0 - x_ratio)
        + wire_delta[6] * (x_ratio)
    )

    pt_delta: NPPointType = f.vector(0.0, 0.0, 0.0)
    pt_delta += (
        face_delta_edge["left"] * (1.0 - x_ratio)
        + face_delta_edge["right"] * (x_ratio)
        + face_delta_edge["front"] * (1.0 - y_ratio)
        + face_delta_edge["back"] * (y_ratio)
        + face_delta_edge["bottom"] * (1.0 - z_ratio)
        + face_delta_edge["top"] * (z_ratio)
    )

    # this corrects the edge calculated delta for the actual face delta
    if face_proj_bool["left"]:
        pt_delta += (face_delta_proj["left"] - 2.0 * face_delta_edge["left"]) * (1.0 - x_ratio)

    if face_proj_bool["right"]:
        pt_delta += (face_delta_proj["right"] - 2.0 * face_delta_edge["right"]) * (x_ratio)

    if face_proj_bool["front"]:
        pt_delta += (face_delta_proj["front"] - 2.0 * face_delta_edge["front"]) * (1.0 - y_ratio)

    if face_proj_bool["back"]:
        pt_delta += (face_delta_proj["back"] - 2.0 * face_delta_edge["back"]) * (y_ratio)

    if face_proj_bool["bottom"]:
        pt_delta += (face_delta_proj["bottom"] - 2.0 * face_delta_edge["bottom"]) * (1.0 - z_ratio)

    if face_proj_bool["top"]:
        pt_delta += (face_delta_proj["top"] - 2.0 * face_delta_edge["top"]) * (z_ratio)

    return pt_delta


def deltas_on_wire(
    mesh: Mesh,
    block_index: int,
    wire_index: int,
    blk: list[NPPointType],
    stlmesh_meshs: dict[str, stlMesh],
    stlmesh_trees: dict[str, KDTree],
    x_ratios: list[float],
    y_ratios: list[float],
    z_ratios: list[float],
) -> list[NPPointType]:
    """Calculates edge position delta at all ratios on this edge"""

    wire = mesh.block_list.blocks[block_index].wire_list[wire_index]
    edge = wire.edge

    if wire.axis == 0:
        t_ratios = x_ratios
    elif wire.axis == 1:
        t_ratios = y_ratios
    else:  # elif (axis == 2)
        t_ratios = z_ratios

    # populate deltas with zeros
    edge_deltas: list[NPPointType] = [f.vector(0.0, 0.0, 0.0) for it in range(len(t_ratios))]

    # collapsed line eithe same index or within position tolerance - return 0's
    if wire.vertices[0].index == wire.vertices[1].index:
        return edge_deltas
    if f.norm(wire.vertices[0].position - wire.vertices[1].position) < constants.TOL:
        return edge_deltas

    # extract deltas from curves
    if edge.kind in ["arc", "origin", "angle"]:
        edge_deltas = edge_deltas_on_arc(mesh, block_index, wire_index, blk, edge_deltas, t_ratios)

    elif edge.kind in ["curve", "spline", "polyline"]:
        edge_deltas = edge_deltas_on_curve(mesh, block_index, wire_index, blk, edge_deltas, t_ratios)

    elif edge.kind in ["project"]:
        edge_deltas = edge_deltas_on_face(
            mesh, block_index, wire_index, blk, edge_deltas, stlmesh_meshs, stlmesh_trees, t_ratios
        )

    return edge_deltas


def edge_deltas_on_arc(
    mesh: Mesh,
    block_index: int,
    wire_index: int,
    blk: list[NPPointType],
    edge_deltas: list[NPPointType],
    t_ratios: list[float],
) -> list[NPPointType]:
    """Returns the edge deltas as result of arc"""

    wire = mesh.block_list.blocks[block_index].wire_list[wire_index]

    for it, t_ratio in enumerate(t_ratios):
        [x_ratio, y_ratio, z_ratio] = hext.wire_to_ratio(wire_index, t_ratio)

        position_on_line = hext.trilinear_interp(blk, x_ratio, y_ratio, z_ratio)

        # TODO
        # note if the blook vertices are projected then the wire.vertices
        # will not be in the right place

        # by using vertices[id] we get the corners in the correct order
        position_on_wire = position_on_arc(
            wire.vertices[0].position,
            wire.edge.third_point.position,
            wire.vertices[1].position,
            position_on_line,
            t_ratio,
        )

        edge_deltas[it] = position_on_wire - position_on_line

    return edge_deltas


def edge_deltas_on_curve(
    mesh: Mesh,
    block_index: int,
    wire_index: int,
    blk: list[NPPointType],
    edge_deltas: list[NPPointType],
    t_ratios: list[float],
) -> list[NPPointType]:
    """Returns the edge deltas as result of curve"""

    wire = mesh.block_list.blocks[block_index].wire_list[wire_index]
    edge = wire.edge

    # check if edge is defined in the correct direction
    reverse = False
    if f.norm(edge.vertex_1.position - wire.vertices[1].position) < constants.TOL:
        reverse = True

    # append vertices to start and end of curve
    _points: NPPointListType
    if edge.vertex_1.position is not None and edge.vertex_2.position is not None:
        _points = np.concatenate(([edge.vertex_1.position], edge.point_array, [edge.vertex_2.position]))
    else:
        _points = edge.point_array

    points = hext.remove_duplicate_points(_points)
    if reverse:
        # points = np.flip(points, 0)
        points.reverse()

    # create params based on projection of points to wire line
    npoints = len(points)
    params = [hext.tratio_on_line(points[0], points[npoints - 1], points[i]) for i in range(npoints)]

    # fit curve function and get point
    if wire.edge.kind.lower() in ["spline", "curve"]:
        try:
            curve_function = scipy.interpolate.make_interp_spline(params, points, check_finite=False)

        except Exception:
            try:
                curve_function = scipy.interpolate.interp1d(
                    params, points, bounds_error=False, fill_value="extrapolate", axis=0  # type: ignore
                )

            except Exception:
                curve_function = None

    else:
        try:
            curve_function = scipy.interpolate.interp1d(
                params, points, bounds_error=False, fill_value="extrapolate", axis=0  # type: ignore
            )

        except Exception:
            curve_function = None

    for it, t_ratio in enumerate(t_ratios):
        [x_ratio, y_ratio, z_ratio] = hext.wire_to_ratio(wire_index, t_ratio)

        position_on_line = hext.trilinear_interp(blk, x_ratio, y_ratio, z_ratio)

        if t_ratio <= 0 or t_ratio >= 1.0:
            position_on_wire = position_on_line

        elif curve_function is None:
            position_on_wire = position_on_line

        else:
            position_on_wire = curve_function(t_ratio)

        edge_deltas[it] = position_on_wire - position_on_line

    return edge_deltas


def edge_deltas_on_face(
    mesh: Mesh,
    block_index: int,
    wire_index: int,
    blk: list[NPPointType],
    edge_deltas: list[NPPointType],
    stlmesh_meshs: dict[str, stlMesh],
    stlmesh_trees: dict[str, KDTree],
    t_ratios: list[float],
) -> list[NPPointType]:
    """Returns the edge deltas as result of face projection"""

    wire = mesh.block_list.blocks[block_index].wire_list[wire_index]
    # edge = wire.edge

    # this edge has a projection - calcultate face deltas
    print(f" Mapping edge projection on edge {wire.edge.description}")

    # save labels to list (remove any duplicates)
    projected_labels: list[str] = []
    if hasattr(wire.edge.data, "label"):
        for label in hext.traverse_list(wire.edge.data.label):
            if label not in projected_labels:
                projected_labels.append(label)

    for it, t_ratio in enumerate(t_ratios):
        [x_ratio, y_ratio, z_ratio] = hext.wire_to_ratio(wire_index, t_ratio)

        if wire.axis == 0:
            projected_option = 6  # search in y and z

        elif wire.axis == 1:
            projected_option = 5  # search in x and z

        else:  # if wire.axis == 2:
            projected_option = 4  # search in x and z

        # projected option 0=nearest point 1=projected alog cell axis
        orient = None
        edge_deltas[it] = delta_on_face(
            mesh,
            block_index,
            blk,
            orient,
            projected_labels,
            projected_option,
            stlmesh_meshs,
            stlmesh_trees,
            x_ratio,
            y_ratio,
            z_ratio,
        )

    return edge_deltas


def face_deltas_on_face(
    mesh: Mesh,
    block_index: int,
    face_index: int,
    orient: OrientType | None,
    blk: list[NPPointType],
    stlmesh_meshs: dict[str, stlMesh],
    stlmesh_trees: dict[str, KDTree],
    x_ratios: list[float],
    y_ratios: list[float],
    z_ratios: list[float],
) -> list[NPPointType]:
    """Returns the delta array as result of face projection across a face"""

    face = mesh.face_list.faces[face_index]

    if orient in ["bottom", "top"]:
        nu_verts = len(x_ratios)
        nv_verts = len(y_ratios)
    elif orient in ["left", "right"]:
        nu_verts = len(y_ratios)
        nv_verts = len(z_ratios)
    elif orient in ["front", "back"]:
        nu_verts = len(z_ratios)
        nv_verts = len(x_ratios)

    # populate deltas with zeros
    face_deltas: list[NPPointType] = []

    # save labels to list (remove any duplicates)
    projected_labels: list[str] = []
    for label in hext.traverse_list(face.label):
        if label not in projected_labels:
            projected_labels.append(label)

    # this face has a projection - calcultate face deltas
    print(f" Mapping face projection on {orient}")

    for iv in range(nv_verts):
        for iu in range(nu_verts):
            if orient in ["left", "right"]:
                projected_option = 1  # projection in x
                x_ratio = 1.0 if orient == "right" else 0.0
                y_ratio = y_ratios[iu]
                z_ratio = z_ratios[iv]

            elif orient in ["front", "back"]:
                projected_option = 2  # projection in y
                x_ratio = x_ratios[iv]
                y_ratio = 1.0 if orient == "back" else 0.0
                z_ratio = z_ratios[iu]

            else:  # if orient in ["bottom","top"]:
                projected_option = 3  # projection in z
                x_ratio = x_ratios[iu]
                y_ratio = y_ratios[iv]
                z_ratio = 1.0 if orient == "top" else 0.0

            # projected option 0=nearest point X=projected alog cell axis
            face_deltas.append(
                delta_on_face(
                    mesh,
                    block_index,
                    blk,
                    orient,
                    projected_labels,
                    projected_option,
                    stlmesh_meshs,
                    stlmesh_trees,
                    x_ratio,
                    y_ratio,
                    z_ratio,
                )
            )

    return face_deltas


def delta_on_face(
    mesh: Mesh,
    block_index: int,
    blk: list[NPPointType],
    orient: OrientType | None,
    projected_labels: list[str],
    projected_option: int,
    stlmesh_meshs: dict[str, stlMesh],
    stlmesh_trees: dict[str, KDTree],
    x_ratio: float | None,
    y_ratio: float | None,
    z_ratio: float | None,
) -> NPPointType:
    """Returns the delta as result of face projection at a single point"""

    # this will also manage the edge and corner projections

    face_delta: NPPointType = f.vector(0.0, 0.0, 0.0)

    [pos_on_block, pos_on_face] = hexp.projection_on_face(
        mesh,
        block_index,
        blk,
        orient,
        projected_labels,
        projected_option,
        stlmesh_meshs,
        stlmesh_trees,
        x_ratio,
        y_ratio,
        z_ratio,
    )

    if pos_on_face is None:
        pos_on_face = pos_on_block

    if pos_on_face is not None:
        face_delta = pos_on_face - pos_on_block

    return face_delta


def position_on_line(
    vertex_1_position: NPPointType, vertex_2_position: NPPointType, t_ratio: float = 0.0
) -> NPPointType:
    """Returns position at ratio t on edge's line"""
    if t_ratio <= 0:
        return vertex_1_position

    if t_ratio >= 1.0:
        return vertex_2_position

    _vector = vertex_2_position - vertex_1_position
    _position = vertex_1_position + t_ratio * _vector

    return _position


def position_on_arc(
    p_start: NPPointType, p_btw: NPPointType, p_end: NPPointType, pos_on_line: NPPointType, t_ratio: float = 0.0
) -> NPPointType:
    """Returns position on arc at point defined by t_ratio"""

    # most of this is take from classy_blocks.util.functions arc_length_3point and arc_mid
    # but modified to return the point on arc at t_ratio

    ### Meticulously transcribed from
    # https://develop.openfoam.com/Development/openfoam/-/blob/master/src/mesh/blockMesh/blockEdges/arcEdge/arcEdge.C

    if t_ratio <= 0:
        return p_start

    if t_ratio >= 1.0:
        return p_end

    vect_a = p_btw - p_start
    vect_b = p_end - p_start

    # Find centre of arcEdge
    asqr = vect_a.dot(vect_a)
    bsqr = vect_b.dot(vect_b)
    adotb = vect_a.dot(vect_b)

    denom = asqr * bsqr - adotb * adotb
    # https://develop.openfoam.com/Development/openfoam/-/blob/master/src/OpenFOAM/primitives/Scalar/floatScalar/floatScalar.H
    if f.norm(denom) < 1e-18:
        # raise ValueError("Invalid arc points!")
        # fustrum example throws this error (third point = p_start) must be bug
        # treat as a straight line
        return pos_on_line

    fact = 0.5 * (bsqr - adotb) / denom

    centre = p_start + 0.5 * vect_a + fact * (np.cross(np.cross(vect_a, vect_b), vect_a))

    # Position vectors from centre
    rad_start = p_start - centre
    # rad_btw = p_btw - centre
    # rad_end = p_end - centre

    mag1 = f.norm(rad_start)
    # mag3 = f.norm(rad_end)

    # The radius from r1 and from r3 will be identical
    # radius = rad_end

    # # Determine the angle
    # angle = np.arccos((rad_start.dot(rad_end)) / (mag1 * mag3))

    # # Check if the vectors define an exterior or an interior arcEdge
    # if np.dot(np.cross(rad_start, rad_btw), np.cross(rad_start, rad_end)) < 0:
    #     angle = 2 * np.pi - angle

    # arc_length = angle * f.norm(radius)

    # vector for radius from center to point on cord
    vect_centre_to_t = pos_on_line - centre

    # vector for point on arc
    t_position = centre + f.unit_vector(vect_centre_to_t) * mag1

    return t_position
