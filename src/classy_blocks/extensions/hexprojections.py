"""functions specifically used by hexmesh for projections"""

import numpy as np
from scipy.spatial import KDTree
from stl import Mesh as stlMesh

from classy_blocks.extensions import hextools as hext
from classy_blocks.items.side import Side
from classy_blocks.mesh import Mesh
from classy_blocks.types import NPPointType, OrientType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


def projection_on_face(
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
) -> list[NPPointType | None]:
    """Returns position on face of block as a result of a projection"""

    [_pos_on_block, _rayx, _rayy, _rayz] = pos_and_rays_on_block(blk, orient, x_ratio, y_ratio, z_ratio)

    # set the initial values
    _pos_on_face: NPPointType | None = None

    # there are 2 lists
    # face_list is list of sides that are projected
    # edge_list is list of edges that are projected
    # projected corner - is currently not catered for

    # we get to here this block and orientation have a face or side projection
    _current_pos_on_block: NPPointType | None = _pos_on_block
    _projection: NPPointType | None = None

    for geom_name in hext.traverse_list(projected_labels):
        # loop rount all the geometry specified on this face - look around here for info
        # https://www.openfoam.com/documentation/guides/v2112/doc/guide-meshing-snappyhexmesh-geometry.html
        if geom_name in mesh.geometry_list.geometry:
            for geom_var in mesh.geometry_list.geometry[geom_name]:
                if geom_var.lower().find("type") != -1:
                    if geom_var.lower().find("trisurfacemesh") != -1:
                        # stl
                        if geom_name in stlmesh_meshs:
                            # stl found in objects dictionary
                            _projection = projection_on_stl(
                                _current_pos_on_block,
                                _rayx,
                                _rayy,
                                _rayz,
                                projected_option,
                                stlmesh_meshs[geom_name],
                                stlmesh_trees[geom_name],
                            )
                        break
                    elif geom_var.lower().find("searchableplane") != -1:
                        # plane
                        _projection = projection_on_shape(
                            "plane",
                            _current_pos_on_block,
                            _rayx,
                            _rayy,
                            _rayz,
                            projected_option,
                            mesh.geometry_list.geometry[geom_name],
                        )
                        break
                    elif geom_var.lower().find("searchableplate") != -1:
                        # platename
                        _projection = projection_on_shape(
                            "plate",
                            _current_pos_on_block,
                            _rayx,
                            _rayy,
                            _rayz,
                            projected_option,
                            mesh.geometry_list.geometry[geom_name],
                        )
                        break
                    elif geom_var.lower().find("searchabledisk") != -1:
                        # disk
                        _projection = projection_on_shape(
                            "disk",
                            _current_pos_on_block,
                            _rayx,
                            _rayy,
                            _rayz,
                            projected_option,
                            mesh.geometry_list.geometry[geom_name],
                        )
                        break
                    elif geom_var.lower().find("searchablebox") != -1:
                        # box
                        _projection = projection_on_shape(
                            "box",
                            _current_pos_on_block,
                            _rayx,
                            _rayy,
                            _rayz,
                            projected_option,
                            mesh.geometry_list.geometry[geom_name],
                        )
                        break
                    elif geom_var.lower().find("searchablerotatedbox") != -1:
                        # rotatedbox
                        _projection = projection_on_shape(
                            "rotatedbox",
                            _current_pos_on_block,
                            _rayx,
                            _rayy,
                            _rayz,
                            projected_option,
                            mesh.geometry_list.geometry[geom_name],
                        )
                        break
                    elif geom_var.lower().find("searchablecylinder") != -1:
                        # cylinder
                        _projection = projection_on_shape(
                            "cylinder",
                            _current_pos_on_block,
                            _rayx,
                            _rayy,
                            _rayz,
                            projected_option,
                            mesh.geometry_list.geometry[geom_name],
                        )
                        break
                    elif geom_var.lower().find("searchablecone") != -1:
                        # cone
                        _projection = projection_on_shape(
                            "cone",
                            _current_pos_on_block,
                            _rayx,
                            _rayy,
                            _rayz,
                            projected_option,
                            mesh.geometry_list.geometry[geom_name],
                        )
                        break
                    elif geom_var.lower().find("searchablesphere") != -1:
                        # sphere
                        _projection = projection_on_sphere(
                            _current_pos_on_block,
                            # _rayx, _rayy, _rayz, projected_option,
                            mesh.geometry_list.geometry[geom_name],
                        )
                        break

            if _projection is not None:
                # check edge projections
                ic = hext.ratio_to_vertex(x_ratio, y_ratio, z_ratio)
                iw = hext.ratio_to_wire(orient, x_ratio, y_ratio, z_ratio)

                if ic is not None:
                    # on corner - check if its projected
                    if geom_name in hext.traverse_list(mesh.block_list.blocks[block_index].vertices[ic].projected_to):
                        # this projection is specified for this corner
                        _pos_on_face = _projection
                        _current_pos_on_block = _projection

                elif iw is not None:
                    # on wire - check if its projected
                    #
                    if mesh.block_list.blocks[block_index].wire_list[iw].edge in mesh.edge_list.edges:
                        if hasattr(mesh.block_list.blocks[block_index].wire_list[iw].edge.data, "label"):
                            # for some reason origin does not have data.label attribute
                            if geom_name in hext.traverse_list(
                                mesh.block_list.blocks[block_index].wire_list[iw].edge.data.label
                            ):
                                # this projection is specified for this edge
                                _pos_on_face = _projection
                                _current_pos_on_block = _projection

                else:
                    # on face - check if this geom is on its projection list
                    for face in mesh.face_list.faces:
                        # face_list is list of all the projected faces
                        # TODO find more elegant way to do this
                        if orient is not None:
                            side = Side(orient, mesh.block_list.blocks[block_index].vertices)
                            if face.side.description == side.description:
                                if geom_name in hext.traverse_list(face.label):
                                    # this projection is specified for this face
                                    _pos_on_face = _projection
                                    _current_pos_on_block = _projection

    return [_pos_on_block, _pos_on_face]


def pos_and_rays_on_block(
    blk: list[NPPointType],
    orient: OrientType | None,
    x_ratio: float | None = 0.0,
    y_ratio: float | None = 0.0,
    z_ratio: float | None = 0.0,
) -> list[NPPointType | None]:
    """Returns position on face of block defined by side and the 3 vectors alignes with the block axis"""

    _pos_on_block: NPPointType | None = None
    # rayx
    _p1 = hext.trilinear_interp(blk, 0.0, y_ratio, z_ratio)
    _p2 = hext.trilinear_interp(blk, 1.0, y_ratio, z_ratio)
    rayx: NPPointType | None = _p2 - _p1
    if orient in ["left", "left"]:
        _pos_on_block = _p1
    elif orient in ["right", "right"]:
        _pos_on_block = _p2

    # rayy
    _p1 = hext.trilinear_interp(blk, x_ratio, 0.0, z_ratio)
    _p2 = hext.trilinear_interp(blk, x_ratio, 1.0, z_ratio)
    rayy: NPPointType | None = _p2 - _p1
    if orient in ["front", "front"]:
        _pos_on_block = _p1
    elif orient in ["back", "back"]:
        _pos_on_block = _p2

    # rayz
    _p1 = hext.trilinear_interp(blk, x_ratio, y_ratio, 0.0)
    _p2 = hext.trilinear_interp(blk, x_ratio, y_ratio, 1.0)
    rayz: NPPointType | None = _p2 - _p1
    if orient in ["bottom", "bottom"]:
        _pos_on_block = _p1
    elif orient in ["top", "top"]:
        _pos_on_block = _p2

    if _pos_on_block is None:
        _pos_on_block = hext.trilinear_interp(blk, x_ratio, y_ratio, z_ratio)

    return [_pos_on_block, rayx, rayy, rayz]


def projection_on_stl(
    pos_on_block: NPPointType | None,
    rayx: NPPointType | None,
    rayy: NPPointType | None,
    rayz: NPPointType | None,
    projected_option: int,
    stlmesh_mesh: stlMesh,
    stlmesh_tree: KDTree,
) -> NPPointType | None:
    """Returns intersection of projected position on face of block and stl surface"""

    # q0 = np.array([0.0,0.0,1.0])
    # q1 = np.array([0.0,0.0,-1.0])
    # p0 = np.array([-1.0,-1.0,0.0])
    # p1 = np.array([1.0,-1.0,0.0])
    # p2 = np.array([0.0,1.0,0.0])
    # _testing = intersect_line_triangle(q0,q1,p0,p1,p2)

    _pos_on_stl: NPPointType | None = None
    _dist_to_stl: float | None = None

    _projection_rayx: NPPointType | None = None
    _dist_to_rayx: float | None = None
    _projection_rayy: NPPointType | None = None
    _dist_to_rayy: float | None = None
    _projection_rayz: NPPointType | None = None
    _dist_to_rayz: float | None = None

    # get list of points in distance order from pos_on_block
    # order = stltree.query_ball_point(pos_on_block,r=constants.VBIG, return_sorted=True)
    if pos_on_block is not None:
        tree_dist, tree_idxs = stlmesh_tree.query(
            pos_on_block, k=3 * stlmesh_mesh.vectors.shape[0], distance_upper_bound=constants.VBIG
        )

    indexes: list[int] = []
    for idx in tree_idxs:  # type: ignore
        new_idx = int(idx / 3)
        if new_idx not in indexes:
            indexes.append(new_idx)

    projection_found = False
    sp_max = len(indexes)

    for sp, ip in enumerate(indexes):
        # find first intersection

        # skip out if projection found
        if projection_found is True and sp > sp_max:
            break

        p0 = stlmesh_mesh[ip][0:3]
        p1 = stlmesh_mesh[ip][3:6]
        p2 = stlmesh_mesh[ip][6:9]

        _projection: NPPointType | None = None
        if rayx is not None:
            if projected_option in [1, 4, 5, 7] and f.norm(rayx) > 0:
                # look along the x axis
                _projection = hext.intersect_line_triangle(pos_on_block, rayx, p0, p1, p2)
                if _projection is not None:
                    if _projection_rayx is not None and _dist_to_rayx is not None:
                        _dist_p = f.norm(_projection - pos_on_block)
                        if _dist_p < _dist_to_rayx:
                            _projection_rayx = _projection
                            _dist_to_rayx = _dist_p
                    else:
                        _projection_rayx = _projection
                        _dist_to_rayx = f.norm(_projection - pos_on_block)

        _projection = None
        if rayy is not None:
            if projected_option in [2, 4, 6, 7] and f.norm(rayy) > 0:
                # look along the y axis
                _projection = hext.intersect_line_triangle(pos_on_block, rayy, p0, p1, p2)
                if _projection is not None:
                    if _projection_rayy is not None:
                        if _projection_rayy is not None and _dist_to_rayy is not None:
                            _dist_p = f.norm(_projection - pos_on_block)
                            if _dist_p < _dist_to_rayy:
                                _projection_rayy = _projection
                                _dist_to_rayy = _dist_p
                    else:
                        _projection_rayy = _projection
                        _dist_to_rayy = f.norm(_projection - pos_on_block)

        _projection = None
        if rayz is not None:
            if projected_option in [3, 5, 6, 7] and f.norm(rayz) > 0:
                # look along the z axis
                _projection = hext.intersect_line_triangle(pos_on_block, rayz, p0, p1, p2)
                if _projection is not None:
                    if _projection_rayz is not None:
                        _dist_p = f.norm(_projection - pos_on_block)
                        if _projection_rayz is not None and _dist_to_rayz is not None:
                            if _dist_p < _dist_to_rayz:
                                _projection_rayz = _projection
                                _dist_to_rayz = _dist_p
                    else:
                        _projection_rayz = _projection
                        _dist_to_rayz = f.norm(_projection - pos_on_block)

        _projection = None
        if projected_option == 0 or projected_option > 3:
            # find nearest point on surface
            _projection = hext.closest_triangle_point(pos_on_block, p0, p1, p2)
            if _projection is not None:
                if _pos_on_stl is not None:
                    # always take the closest point to block
                    _dist_p = f.norm(_projection - pos_on_block)
                    if _dist_p is not None and _dist_to_stl is not None:
                        if _dist_p < _dist_to_stl:
                            _pos_on_stl = _projection
                            _dist_to_stl = _dist_p
                else:
                    _pos_on_stl = _projection
                    _dist_to_stl = f.norm(_projection - pos_on_block)

        if (
            _projection_rayz is not None
            or _projection_rayy is not None
            or _projection_rayz is not None
            or _pos_on_stl is not None
        ):
            if projection_found is False:
                # search the next 10 patches then skip out
                projection_found = True
                sp_max = min(sp + 10, len(indexes))

    # prioritise ray intersection
    if _dist_to_rayx is not None or _dist_to_rayy is not None or _dist_to_rayz is not None:
        _dists = [_dist_to_rayx, _dist_to_rayy, _dist_to_rayz]
        _ray_min_dist = min(x for x in _dists if x is not None)
        if _dist_to_stl is None or _ray_min_dist < 2.0 * _dist_to_stl:
            _dist_to_stl = _ray_min_dist
            if _ray_min_dist == _dist_to_rayz:
                _pos_on_stl = _projection_rayz
            elif _ray_min_dist == _dist_to_rayy:
                _pos_on_stl = _projection_rayy
            else:
                _pos_on_stl = _projection_rayx

    return _pos_on_stl


def projection_on_sphere(
    pos_on_block: NPPointType | None,
    # rayx:NPPointType,
    # rayy:NPPointType,
    # rayz:NPPointType,
    # projected_option:int,
    geometry: list[str],
) -> NPPointType | None:
    """Returns projection onto a projected sphere"""

    #
    # Geometry Format
    # sphere
    # {
    #     type            searchableSphere;
    #     centre          (0 0 0);
    #     radius          3;
    # }

    # extract specification from geometry list
    projection: NPPointType | None = pos_on_block
    # origin = None
    center: NPPointType | None = None
    radius: float | None = None
    for geom_var in geometry:
        # if (geom_var.lower().find("origin") != -1  ):
        #    # Extract Numbers in Brackets in String
        #    _ords = hext.numbers_in_string(geom_var)
        #    if (len(_ords)>=3) :
        #        origin = f.vector(_ords[0],_ords[1],_ords[2])
        if geom_var.lower().find("centre") != -1 or geom_var.lower().find("center") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                center = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("radius") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 1:
                radius = _ords[0]

    if center is not None and radius is not None:
        # what is the origin supposed to do ?
        # note that a all calls the projection is to the surface of the sphere
        # irrespective of the rayx y and z
        # (ie it does not only work for the edge)

        center_to_block = pos_on_block - center

        projection = center + f.unit_vector(center_to_block) * radius

    return projection


def projection_on_shape(
    shape_name: str,
    pos_on_block: NPPointType | None,
    rayx: NPPointType | None,
    rayy: NPPointType | None,
    rayz: NPPointType | None,
    projected_option: int,
    geometry: list[str],
) -> NPPointType | None:
    """Returns projection on a shape"""

    projection: NPPointType | None = pos_on_block

    # get parameters from geometry
    if shape_name == "plane":
        [basepoint, normal] = get_plane_geometry(geometry)
        if basepoint is None or normal is None:
            return projection

    elif shape_name == "plate":
        [origin, span] = get_plate_geometry(geometry)
        if origin is None or span is None:
            return projection

    elif shape_name == "disk":
        (origin, normal, radius) = get_disk_geometry(geometry)
        if origin is None or normal is None or radius is None:
            return projection

    elif shape_name == "box":
        [box_min, box_max] = get_box_geometry(geometry)
        if box_min is None or box_max is None:
            return projection

    elif shape_name == "rotatedbox":
        [span, origin, e1, e3] = get_rotatedbox_geometry(geometry)
        if span is None or origin is None or e1 is None or e3 is None:
            return projection

    elif shape_name == "cylinder":
        (point1, point2, radius) = get_cylinder_geometry(geometry)
        if point1 is None or point2 is None:
            return projection

    elif shape_name == "cone":
        (point1, radius1, inner_radius1, point2, radius2, inner_radius2) = get_cone_geometry(geometry)
        if (
            point1 is None
            or point2 is None
            or radius1 is None
            or radius2 is None
            or inner_radius1 is None
            or inner_radius2 is None
        ):
            return projection

    # search shape for projection
    _projection: NPPointType | None = None
    _projection_rayx: NPPointType | None = None
    _dist_to_rayx: float | None = None
    if projected_option in [1, 4, 5, 7] and rayx is not None:
        if hext.norm(rayx) > 0:
            # look along the x axis
            if shape_name == "plane":
                _projection = hext.intersect_line_plane(pos_on_block, rayx, basepoint, normal)
            elif shape_name == "plate":
                pass
            elif shape_name == "disk":
                pass
            elif shape_name == "box":
                pass
            elif shape_name == "rotatedbox":
                pass
            elif shape_name == "cylinder":
                pass
            elif shape_name == "cone":
                pass

            if _projection is not None:
                _projection_rayx = _projection
                _dist_to_rayx = f.norm(_projection - pos_on_block)

    _projection = None
    _projection_rayy: NPPointType | None = None
    _dist_to_rayy: float | None = None
    if projected_option in [2, 4, 6, 7] and rayy is not None:
        if hext.norm(rayy) > 0:
            # look along the y axis
            if shape_name == "plane":
                _projection = hext.intersect_line_plane(pos_on_block, rayy, basepoint, normal)
            elif shape_name == "plate":
                pass
            elif shape_name == "disk":
                pass
            elif shape_name == "box":
                pass
            elif shape_name == "rotatedbox":
                pass
            elif shape_name == "cylinder":
                pass
            elif shape_name == "cone":
                pass

            if _projection is not None:
                _projection_rayy = _projection
                _dist_to_rayy = f.norm(_projection - pos_on_block)

    _projection = None
    _projection_rayz: NPPointType | None = None
    _dist_to_rayz: float | None = None
    if projected_option in [3, 5, 6, 7] and rayz is not None:
        if hext.norm(rayz) > 0:
            # look along the z axis
            if shape_name == "plane":
                _projection = hext.intersect_line_plane(pos_on_block, rayz, basepoint, normal)
            elif shape_name == "plate":
                pass
            elif shape_name == "disk":
                pass
            elif shape_name == "box":
                pass
            elif shape_name == "rotatedbox":
                pass
            elif shape_name == "cylinder":
                pass
            elif shape_name == "cone":
                pass

            if _projection is not None:
                _projection_rayz = _projection
                _dist_to_rayz = f.norm(_projection - pos_on_block)

    _projection = None
    _projection_closest: NPPointType | None = None
    _dist_to_closest: float | None = None
    if projected_option == 0 or projected_option > 3:
        # find nearest point on surface
        if shape_name == "plane":
            pass
            # _projection = hext.closest_point_on_plane(pos_on_block,basepoint,normal)
        elif shape_name == "plate":
            pass
        elif shape_name == "disk":
            pass
        elif shape_name == "box":
            pass
        elif shape_name == "rotatedbox":
            pass
        elif shape_name == "cylinder":
            pass
        elif shape_name == "cone":
            pass

        if _projection is not None:
            _projection_closest = _projection
            _dist_to_closest = f.norm(_projection - pos_on_block)

    # prioritise ray intersection
    if _dist_to_rayx is not None or _dist_to_rayy is not None or _dist_to_rayz is not None:
        _dists = [_dist_to_rayx, _dist_to_rayy, _dist_to_rayz]
        _ray_min_dist = min(x for x in _dists if x is not None)
        if _dist_to_closest is None or _ray_min_dist < 2.0 * _dist_to_closest:
            _dist_to_projection = _ray_min_dist
            if _ray_min_dist == _dist_to_rayz:
                projection = _projection_rayz
            elif _ray_min_dist == _dist_to_rayy:
                projection = _projection_rayy
            else:
                projection = _projection_rayx
        else:
            projection = _projection_closest

    if projection is None:
        projection = pos_on_block

    return projection


def get_plane_geometry(geometry: list[str]) -> list[NPPointType | None]:
    """Returns basepoint and normal from geometry dictionary"""

    # Geometry Format
    # Plane defined by point and a normal vector:
    # plane
    # {
    #     type            searchablePlane;
    #     planeType       pointAndNormal;
    #     pointAndNormalDict
    #     {
    #         basePoint       (1 1 1);
    #         normal          (0 1 0);
    #     }
    # }
    # Plane defined by 3 points on the plane:
    # plane
    # {
    #     type            searchablePlane;
    #     planeType       embeddedPoints;

    #     embeddedPointsDict
    #     {
    #         point1          (1 1 1);
    #         point2          (0 1 0);
    #         point3          (0 0 1)
    #     }
    # }
    # Plane defined by plane equation:
    # plane
    # {
    #     type            searchablePlane;
    #     planeType       planeEquation;
    #     planeEquationDict
    #     {
    #         a  0;
    #         b  0;
    #         c  1; // to create plane with normal towards +z direction ...
    #         d  2; // ... at coordinate: z = 2
    #     }
    # }

    # extract specification from geometry list
    # ideally this would be held as values not strings in geometry_list

    p_type: int | None = None
    basepoint: NPPointType | None = None
    normal: NPPointType | None = None
    point1: NPPointType | None = None
    point2: NPPointType | None = None
    point3: NPPointType | None = None
    acoef: float | None = None
    bcoef: float | None = None
    ccoef: float | None = None
    dcoef: float | None = None

    for geom_var in geometry:
        if geom_var.lower().find("pointandnormal") != -1:
            # point and nornal formal
            p_type = 1
            for gvar in geometry:
                if (gvar.lower().find("basepoint") != -1 or gvar.lower().find("point") != -1) and basepoint is None:
                    # Extract Numbers in Brackets in String
                    _ords = hext.numbers_in_string(gvar)
                    if len(_ords) >= 3:
                        basepoint = f.vector(_ords[0], _ords[1], _ords[2])
                if gvar.lower().find("normal") != -1 and normal is None:
                    # Extract Numbers in Brackets in String
                    _ords = hext.numbers_in_string(gvar)
                    if len(_ords) >= 3:
                        normal = f.vector(_ords[0], _ords[1], _ords[2])
            break

        if geom_var.lower().find("embeddedpoints") != -1:
            # embedded points format
            p_type = 2
            for gvar in geometry:
                if gvar.lower().find("point1") != -1 and point1 is None:
                    # Extract Numbers in Brackets in String
                    _ords = hext.numbers_in_string(gvar)
                    if len(_ords) >= 3:
                        point1 = f.vector(_ords[0], _ords[1], _ords[2])
                if gvar.lower().find("point2") != -1 and point2 is None:
                    # Extract Numbers in Brackets in String
                    _ords = hext.numbers_in_string(gvar)
                    if len(_ords) >= 3:
                        point2 = f.vector(_ords[0], _ords[1], _ords[2])
                if gvar.lower().find("point3") != -1 and point3 is None:
                    # Extract Numbers in Brackets in String
                    _ords = hext.numbers_in_string(gvar)
                    if len(_ords) >= 3:
                        point3 = f.vector(_ords[0], _ords[1], _ords[2])
            break

        if geom_var.lower().find("planeequation") != -1:
            # plane equation format
            # acoef x + bcoef y + ccoef z = dcoef
            p_type = 3
            for gvar in geometry:
                if gvar.lower().find(" a ") != -1 and acoef is None:
                    # Extract Numbers in Brackets in String
                    _ords = hext.numbers_in_string(gvar)
                    if len(_ords) >= 1:
                        acoef = _ords[0]
                if gvar.lower().find(" b ") != -1 and bcoef is None:
                    # Extract Numbers in Brackets in String
                    _ords = hext.numbers_in_string(gvar)
                    if len(_ords) >= 1:
                        bcoef = _ords[0]
                if gvar.lower().find(" c ") != -1 and ccoef is None:
                    # Extract Numbers in Brackets in String
                    _ords = hext.numbers_in_string(gvar)
                    if len(_ords) >= 1:
                        ccoef = _ords[0]
                if gvar.lower().find(" d ") != -1 and dcoef is None:
                    # Extract Numbers in Brackets in String
                    _ords = hext.numbers_in_string(gvar)
                    if len(_ords) >= 1:
                        dcoef = _ords[0]
            break

    if p_type is None or p_type not in [1, 2, 3]:
        return [None, None]

    if p_type == 1 and (basepoint is None or normal is None):
        return [None, None]

    if p_type == 2:
        # convert to point and normal
        if point1 is None or point2 is None or point3 is None:
            return [None, None]
        else:
            basepoint = point1
            normal = np.cross(point2 - point1, point3 - point1)

    if p_type == 3:
        # convert to point and normal
        if acoef is None or bcoef is None or ccoef is None or dcoef is None:
            return [None, None]
        else:
            mag_coef2 = acoef * acoef + bcoef * bcoef + ccoef * ccoef
            if mag_coef2 == 0:
                return [None, None]
            mag_coef = np.sqrt(mag_coef2)
            basepoint = f.vector(acoef * dcoef / mag_coef2, bcoef * dcoef / mag_coef2, ccoef * dcoef / mag_coef2)
            normal = f.vector(acoef / mag_coef, bcoef / mag_coef, ccoef / mag_coef)

    return [basepoint, normal]


def get_plate_geometry(geometry: list[str]) -> list[NPPointType | None]:
    """Returns pate origin and span from geometry dictionary"""

    # Geometry Format
    # plate
    # {
    #     type            searchablePlate;
    #     origin          (0 0 0);
    #     span            (2 4 0);
    # }

    origin: NPPointType | None = None
    span: NPPointType | None = None

    for geom_var in geometry:
        if geom_var.lower().find("origin") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                origin = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("normal") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                if _ords[0] == 0 or _ords[1] == 0 or _ords[3] == 0:
                    span = f.vector(_ords[0], _ords[1], _ords[2])

    if origin is None or span is None:
        return [None, None]

    return [origin, span]


def get_disk_geometry(geometry: list[str]) -> tuple[NPPointType | None, NPPointType | None, float | None]:
    """Returns disk origin normal and radius from geometry dictionary"""

    # Geometry Format
    # disk
    # {
    #     type            searchableDisk;
    #     origin          (0 0 0);
    #     normal          (0 1 0);
    #     radius          0.314;
    # }

    origin: NPPointType | None = None
    normal: NPPointType | None = None
    radius: float | None = None

    for geom_var in geometry:
        if geom_var.lower().find("origin") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                origin = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("normal") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                normal = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("radius") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 1:
                radius = _ords[0]

    if origin is None or normal is None or radius is None:
        return (None, None, None)

    return (origin, normal, radius)


def get_box_geometry(geometry: list[str]) -> list[NPPointType | None]:
    """Returns box min and max from geometry dictionary"""

    # Geometry Format
    # box1
    # {
    #     type            searchableBox;
    #     min             (0 0 0);
    #     max             (1 1 1);
    # }

    box_min: NPPointType | None = None
    box_max: NPPointType | None = None

    for geom_var in geometry:
        if geom_var.lower().find("min") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                box_min = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("max") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                box_max = f.vector(_ords[0], _ords[1], _ords[2])

    if box_max is None or box_min is None:
        return [None, None]

    return [box_min, box_max]


def get_rotatedbox_geometry(geometry: list[str]) -> list[NPPointType | None]:
    """Returns box min and max from geometry dictionary"""

    # Geometry Format
    # boxRotated
    # {
    #     type            searchableRotatedBox;
    #     span            (5 4 3);
    #     origin          (0 0 0);
    #     e1              (1 0.5 0);
    #     e3              (0 0.5 1);
    # }
    span: NPPointType | None = None
    origin: NPPointType | None = None
    e1: NPPointType | None = None
    e3: NPPointType | None = None

    for geom_var in geometry:
        if geom_var.lower().find("span") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                span = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("origin") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                origin = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("e1") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                e1 = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("e3") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                e3 = f.vector(_ords[0], _ords[1], _ords[2])

    if span is None or origin is None or e1 is None or e3 is None:
        return [None, None, None, None]

    return [span, origin, e1, e3]


def get_cylinder_geometry(geometry: list[str]) -> tuple[NPPointType | None, NPPointType | None, float | None]:
    """Returns cylinder point1 point2 and radius from geometry dictionary"""

    # Geometry Format
    # cylinder
    # {
    #     type            searchableCylinder;
    #     point1          (1.5 1 -0.5);
    #     point2          (3.5 2 0.5);
    #     radius          0.05;}

    point1: NPPointType | None = None
    point2: NPPointType | None = None
    radius: float | None = None

    for geom_var in geometry:
        if geom_var.lower().find("point1") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                point1 = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("point2") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                point2 = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("radius") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 1:
                radius = _ords[0]

    if point1 is None or point2 is None or radius is None:
        return (None, None, None)

    return (point1, point2, radius)


def get_cone_geometry(
    geometry: list[str],
) -> tuple[NPPointType | None, float | None, float | None, NPPointType | None, float | None, float | None]:
    """Returns cone point1 radius1 innerradius1 point2 radius2 innerradius2 from geometry dictionary"""

    # Geometry Format
    # cone
    # {
    #     type            searchableCone;
    #     point1          (0 0 0);
    #     radius1         1.5;
    #     innerRadius1    0.25;
    #     point2          (10 0 0);
    #     radius2         3.0;
    #     innerRadius2    1.0;
    # }

    point1 = None
    radius1 = None
    inner_radius1 = None
    point2 = None
    radius2 = None
    inner_radius2 = None

    for geom_var in geometry:
        if geom_var.lower().find("point1") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                point1 = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("point2") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 3:
                point2 = f.vector(_ords[0], _ords[1], _ords[2])
        if geom_var.lower().find("radius1") != -1 or geom_var.lower().find("radius_1") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 1:
                radius1 = _ords[0]
        if geom_var.lower().find("radius2") != -1 or geom_var.lower().find("radius_2") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 1:
                radius2 = _ords[0]
        if geom_var.lower().find("innerradius1") != -1 or geom_var.lower().find("inner_radius_1") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 1:
                inner_radius1 = _ords[0]
        if geom_var.lower().find("innerradius2") != -1 or geom_var.lower().find("inner_radius_2") != -1:
            # Extract Numbers in Brackets in String
            _ords = hext.numbers_in_string(geom_var)
            if len(_ords) >= 1:
                inner_radius2 = _ords[0]

    if (
        point1 is None
        or point2 is None
        or radius1 is None
        or radius2 is None
        or inner_radius1 is None
        or inner_radius2 is None
    ):
        return (None, None, None, None, None, None)

    return (point1, radius1, inner_radius1, point2, radius2, inner_radius2)
