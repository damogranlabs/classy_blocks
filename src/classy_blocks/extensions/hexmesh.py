"""Creates the Hexmesh of the mesh from the classy_blocks objects."""

# The approach is very similar to that published in the paper
# Geometry controlled refinement for hexahedra:
# Elsweijer, Sandro and Holke, Johannes and Kleinert, Jan and Reith, Dirk (2022)
# Constructing a Volume Geometry Map for Hexahedra with Curved Boundary Geometries.
# In: SIAM International Meshing Roundtable Workshop 2022.
# SIAM International Meshing Roundtable Workshop 2022, 22. - 25. Feb. 2022,
#
# BUT THIS IMPLEMENTATION IS MUCH MUCH SIMPLER TO UNDERSTAND AS I CREATED IT AND
# THEN FOUND THE ABOVE REFERENCE AFTER THE CODE WAS COMPLETE !!!

import time as time

import numpy as np
from scipy.spatial import KDTree

from classy_blocks import Mesh
from classy_blocks.extensions import hexdeltas as hexd
from classy_blocks.extensions import hexgrading as hexg
from classy_blocks.extensions import hextools as hext
from classy_blocks.extensions.hexcell import HexCell
from classy_blocks.extensions.hexcell_list import HexCellList
from classy_blocks.extensions.hexconstants import FRONT_TO_BACK, RIGHT_TO_LEFT, TOP_TO_BOTTOM
from classy_blocks.extensions.hexreader import read_geometry
from classy_blocks.extensions.hexvertex import HexVertex
from classy_blocks.extensions.hexvertex_list import HexVertexList
from classy_blocks.items.side import Side
from classy_blocks.items.vertex import Vertex
from classy_blocks.modify.cell import Cell
from classy_blocks.types import NPPointType, OrientType
from classy_blocks.util import constants
from classy_blocks.util import functions as f


class HexMesh:
    """generates HexCells from the classy_blocks Mesh class object"""

    def __init__(self, mesh: Mesh | None = None, quality_metrics: bool = False, add_neighbours: bool = False) -> None:
        # List of all added operations/shapes

        # initiate lists
        self.clear_mesh()

        if mesh is not None:
            # make sure mesh is assembled
            if not mesh.is_assembled:
                mesh.assemble()
            # update gradings if they are not already defined
            mesh.block_list.propagate_gradings()

            # generate mesh
            self.add_mesh(mesh, quality_metrics=quality_metrics, add_neighbours=add_neighbours)

    def add_mesh(self, mesh: Mesh | None, quality_metrics: bool = False, add_neighbours: bool = False) -> None:
        """Generate hexcells from classy_blocks mesh"""

        # generates hex mesh from mesh.blocks
        # using a trilinear interpolation to get the corners of the hex
        # using the simple alogrithim here - https://paulbourke.net/miscellaneous/interpolation/
        # than add the face/edges correction to correct for edge curves and projections
        # (this is a really crude way of implementing a jacobian at each node)

        if mesh is None:
            return
        self.mesh: Mesh = mesh
        self.quality_metrics = quality_metrics
        _start = time.time()
        print("Creating Hexmesh")

        # first check if we need to read any files specified by geometrty
        [stlmesh_meshs, stlmesh_trees] = read_geometry(self.mesh)

        print(f" Total no of blocks = {len(self.mesh.block_list.blocks)}")
        _running_total_cells = 0
        _nprint = 1000

        # holding list for the cell indices in this block
        block_cells: list[Cell] = []
        block_face_vertex_idx: dict[int, list[int]] = {}
        block_face_vertex_pos: list[NPPointType] = []
        block_face_tree: dict[int, KDTree] = {}
        iw: int | None = None

        for block_index, block in enumerate(self.mesh.block_list.blocks):
            # loop around each block to generate cells
            # had a random error with this call when mesh.write not called !!
            _description = block.description
            nchars = len(_description) - 1
            print(f" Description {_description[:nchars]}")

            # create a cell of this block
            block_cells.append(Cell(block))
            # add neighbours to this cell
            for ib in range(block_index):
                # add any neighbours already created
                block_cells[block_index].add_neighbour(block_cells[ib])

            # initialise cell index holding array
            block_hexcell_list: list[int] = []
            block_face_vertex_pos = []
            block_face_vertex_idx[block_index] = []

            # calculate x y and z ratios for the grading
            z_ratios = hexg.grading_ratios(grading=block.axes[2].grading)
            y_ratios = hexg.grading_ratios(grading=block.axes[1].grading)
            x_ratios = hexg.grading_ratios(grading=block.axes[0].grading)

            nx_cells = len(x_ratios) - 1
            ny_cells = len(y_ratios) - 1
            nz_cells = len(z_ratios) - 1

            # extract the corners from the vertices
            blk: list[NPPointType] = []
            for iv in range(len(block.vertices)):
                _pt = np.array(
                    [block.vertices[iv].position[0], block.vertices[iv].position[1], block.vertices[iv].position[2]],
                    dtype=constants.DTYPE,
                )
                blk.append(_pt)

            # check for projected corners and move the blk positions if projected

            deltap: list[NPPointType | None] = []
            for iv, vertex in enumerate(block.vertices):
                if len(vertex.projected_to) > 0:
                    # save labels to list (remove any duplicates)
                    projected_labels: list[str] = []
                    for label in hext.traverse_list(vertex.projected_to):
                        if label not in projected_labels:
                            projected_labels.append(label)

                    # this corner has a projection - calcultate face deltas and move vertices
                    print(f" Mapping corner projection on vertex {vertex.description}")

                    x_ratio = 1.0 if (iv in [1, 2, 5, 6]) else 0.0
                    y_ratio = 1.0 if (iv in [3, 2, 6, 7]) else 0.0
                    z_ratio = 1.0 if (iv in [4, 5, 6, 7]) else 0.0

                    # projected option 0=nearest point 1=projected alog cell axis
                    projected_option = 7
                    orient = None
                    _delta_face = hexd.delta_on_face(
                        self.mesh,
                        block_index,
                        blk,
                        orient,
                        projected_labels,
                        projected_option,
                        stlmesh_meshs,  # type: ignore
                        stlmesh_trees,  # type: ignore
                        x_ratio,
                        y_ratio,
                        z_ratio,
                    )
                else:
                    _delta_face = None
                deltap.append(_delta_face)

            # move the vertices
            for iv in range(len(block.vertices)):
                if deltap[iv] is not None:
                    blk[iv] = blk[iv] + deltap[iv]

            # now populate the edge_deltas dictionary for all wires
            edge_deltas: dict[int, list[NPPointType]] = {}
            for iw in range(len(block.wire_list)):
                edge_deltas[iw] = hexd.deltas_on_wire(
                    self.mesh,
                    block_index,
                    iw,
                    blk,
                    stlmesh_meshs,  # type: ignore
                    stlmesh_trees,  # type: ignore
                    x_ratios,
                    y_ratios,
                    z_ratios,
                )

            # check for face projection and save it
            projected_face_deltas: dict[OrientType, list[NPPointType]] = {}
            for face_index, face in enumerate(self.mesh.face_list.faces):
                # face_list is list of all the projected faces
                for orient in constants.FACE_MAP.keys():
                    side = Side(orient, block.vertices)
                    if face.side.description == side.description:
                        # this block and orient have a face projection
                        projected_face_deltas[orient] = hexd.face_deltas_on_face(
                            self.mesh,
                            block_index,
                            face_index,
                            orient,
                            blk,
                            stlmesh_meshs,  # type: ignore
                            stlmesh_trees,  # type: ignore
                            x_ratios,
                            y_ratios,
                            z_ratios,
                        )

            # initialise delta lists
            wire_delta: list[NPPointType] = [f.vector(0, 0, 0) for _ in range(12)]
            face_delta_edge: dict[OrientType, NPPointType] = {}
            face_delta_proj: dict[OrientType, NPPointType] = {}
            face_proj_bool: dict[OrientType, bool] = {}

            for iz in range(nz_cells):
                # zaxis loop

                for iy in range(ny_cells):
                    # y axis loop

                    for ix in range(nx_cells):
                        # x axis loop

                        # set neighbour cell_index pointers
                        left_cell_index = (
                            block_hexcell_list[-1] if (len(block_hexcell_list) > 0 and x_ratios[ix] != 0.0) else None
                        )
                        front_cell_index = (
                            block_hexcell_list[-nx_cells]
                            if (len(block_hexcell_list) >= nx_cells and y_ratios[iy] != 0.0)
                            else None
                        )
                        bottom_cell_index = (
                            block_hexcell_list[-nx_cells * ny_cells]
                            if (len(block_hexcell_list) >= nx_cells * ny_cells and z_ratios[iz] != 0.0)
                            else None
                        )

                        # load coordinates to vertex and vertex_list.add should return unique vertex id
                        cell_vertices: list[HexVertex] = []
                        for iv in range(8):
                            # first try and copy vertices from surrounding cells
                            if left_cell_index is not None and iv in [0, 3, 7, 4]:
                                cell_vertices.append(
                                    self.hexcell_list.hexcells[left_cell_index].hexvertices[RIGHT_TO_LEFT[iv]]
                                )

                            elif front_cell_index is not None and iv in [0, 1, 5, 4]:
                                cell_vertices.append(
                                    self.hexcell_list.hexcells[front_cell_index].hexvertices[FRONT_TO_BACK[iv]]
                                )

                            elif bottom_cell_index is not None and iv in [0, 1, 2, 3]:
                                cell_vertices.append(
                                    self.hexcell_list.hexcells[bottom_cell_index].hexvertices[TOP_TO_BOTTOM[iv]]
                                )

                            else:
                                # no vertex to copy - calculate new vertex

                                # calculate interpolated points from the appropriate face
                                x_rat = x_ratios[ix] if iv in [0, 3, 4, 7] else x_ratios[ix + 1]
                                y_rat = y_ratios[iy] if iv in [0, 1, 4, 5] else y_ratios[iy + 1]
                                z_rat = z_ratios[iz] if iv in [0, 1, 2, 3] else z_ratios[iz + 1]

                                wire_delta[0] = (
                                    edge_deltas[0][x_ratios.index(x_ratios[ix + 1])]
                                    if iv in [1, 2, 5, 6]
                                    else edge_deltas[0][x_ratios.index(x_ratios[ix])]
                                )

                                wire_delta[1] = (
                                    edge_deltas[1][x_ratios.index(x_ratios[ix + 1])]
                                    if iv in [1, 2, 5, 6]
                                    else edge_deltas[1][x_ratios.index(x_ratios[ix])]
                                )

                                wire_delta[2] = (
                                    edge_deltas[2][x_ratios.index(x_ratios[ix + 1])]
                                    if iv in [1, 2, 5, 6]
                                    else edge_deltas[2][x_ratios.index(x_ratios[ix])]
                                )

                                wire_delta[3] = (
                                    edge_deltas[3][x_ratios.index(x_ratios[ix + 1])]
                                    if iv in [1, 2, 5, 6]
                                    else edge_deltas[3][x_ratios.index(x_ratios[ix])]
                                )

                                wire_delta[4] = (
                                    edge_deltas[4][y_ratios.index(y_ratios[iy + 1])]
                                    if iv in [3, 2, 6, 7]
                                    else edge_deltas[4][y_ratios.index(y_ratios[iy])]
                                )

                                wire_delta[5] = (
                                    edge_deltas[5][y_ratios.index(y_ratios[iy + 1])]
                                    if iv in [3, 2, 6, 7]
                                    else edge_deltas[5][y_ratios.index(y_ratios[iy])]
                                )

                                wire_delta[6] = (
                                    edge_deltas[6][y_ratios.index(y_ratios[iy + 1])]
                                    if iv in [3, 2, 6, 7]
                                    else edge_deltas[6][y_ratios.index(y_ratios[iy])]
                                )

                                wire_delta[7] = (
                                    edge_deltas[7][y_ratios.index(y_ratios[iy + 1])]
                                    if iv in [3, 2, 6, 7]
                                    else edge_deltas[7][y_ratios.index(y_ratios[iy])]
                                )

                                wire_delta[8] = (
                                    edge_deltas[8][z_ratios.index(z_ratios[iz + 1])]
                                    if iv in [4, 5, 6, 7]
                                    else edge_deltas[8][z_ratios.index(z_ratios[iz])]
                                )

                                wire_delta[9] = (
                                    edge_deltas[9][z_ratios.index(z_ratios[iz + 1])]
                                    if iv in [4, 5, 6, 7]
                                    else edge_deltas[9][z_ratios.index(z_ratios[iz])]
                                )

                                wire_delta[10] = (
                                    edge_deltas[10][z_ratios.index(z_ratios[iz + 1])]
                                    if iv in [4, 5, 6, 7]
                                    else edge_deltas[10][z_ratios.index(z_ratios[iz])]
                                )

                                wire_delta[11] = (
                                    edge_deltas[11][z_ratios.index(z_ratios[iz + 1])]
                                    if iv in [4, 5, 6, 7]
                                    else edge_deltas[11][z_ratios.index(z_ratios[iz])]
                                )

                                # get the point on the block
                                hex_vert = hext.trilinear_interp(blk, x_rat, y_rat, z_rat)

                                # allow for projections
                                for orient in constants.FACE_MAP.keys():
                                    face_proj_bool[orient] = False
                                    face_delta_edge[orient] = f.vector(0.0, 0.0, 0.0)
                                    face_delta_proj[orient] = f.vector(0.0, 0.0, 0.0)
                                    if len(projected_face_deltas) > 0:
                                        # skip vertices
                                        ic = hext.ratio_to_vertex(x_rat, y_rat, z_rat)
                                        if ic is None:
                                            if orient in projected_face_deltas.keys():
                                                # skip edges
                                                iw = hext.ratio_to_wire(orient, x_rat, y_rat, z_rat)
                                                if iw is None:
                                                    if orient in ["left", "right"]:
                                                        idx = z_ratios.index(z_rat) * (ny_cells + 1) + y_ratios.index(
                                                            y_rat
                                                        )
                                                    elif orient in ["bottom", "top"]:
                                                        idx = y_ratios.index(y_rat) * (nx_cells + 1) + x_ratios.index(
                                                            x_rat
                                                        )
                                                    elif orient in ["front", "back"]:
                                                        idx = x_ratios.index(x_rat) * (nz_cells + 1) + z_ratios.index(
                                                            z_rat
                                                        )
                                                    face_delta_proj[orient] = projected_face_deltas[orient][idx]
                                                    face_proj_bool[orient] = True

                                # get point delta from face deltas
                                hex_delta = hexd.point_delta(
                                    wire_delta, face_delta_edge, face_delta_proj, face_proj_bool, x_rat, y_rat, z_rat
                                )

                                hex_position = hex_vert + hex_delta

                                dup_vertex = None
                                if (
                                    x_rat == 0.0
                                    or x_rat == 1.0
                                    or y_rat == 0.0
                                    or y_rat == 1.0
                                    or z_rat == 0.0
                                    or z_rat == 1.0
                                ):
                                    # check for duplicate vertex in any already defined neighbour blocks
                                    for neighbour in block_cells[block_index].neighbours.values():
                                        if neighbour is not None:
                                            dup_list = block_face_tree[block_cells.index(neighbour)].query_ball_point(
                                                hex_position, r=constants.TOL, workers=-1
                                            )
                                            if len(dup_list) > 0:
                                                dup_index = block_face_vertex_idx[block_cells.index(neighbour)][
                                                    dup_list[0]
                                                ]
                                                dup_vertex = self.hexvertex_list.hexvertices[dup_index]
                                                break
                                            else:
                                                continue
                                            break
                                        else:
                                            continue
                                        break

                                if dup_vertex is not None:
                                    # add the duplicate
                                    cell_vertices.append(dup_vertex)
                                else:
                                    # add new vertex
                                    cell_vertices.append(self.hexvertex_list.add(HexVertex(hex_position, iv)))

                            # if on face add vertex to face list
                            if (
                                x_rat == 0.0
                                or x_rat == 1.0
                                or y_rat == 0.0
                                or y_rat == 1.0
                                or z_rat == 0.0
                                or z_rat == 1.0
                            ):
                                block_face_vertex_idx[block_index].append(cell_vertices[iv].index)
                                block_face_vertex_pos.append(cell_vertices[iv].position)

                        # create cell
                        _hexcell = HexCell(block, cell_vertices)
                        self.hexcell_list.add(_hexcell)

                        if len(self.hexcell_list.hexcells) == 1 or len(self.hexcell_list.hexcells) % _nprint == 0:
                            print(f" No of cells = {len(self.hexcell_list.hexcells)}", end="\r")

                        block_hexcell_list.append(_hexcell.index)

            # build the KDtree of the face vertex positions
            block_face_tree[block_index] = KDTree(block_face_vertex_pos)  # type: ignore

            print(
                f" Block cells = "
                f"{len(self.hexcell_list.hexcells)-_running_total_cells} of "
                f"{len(self.hexcell_list.hexcells)}"
            )
            _running_total_cells = len(self.hexcell_list.hexcells)

            # add neighbours to cells for this block "without face checks"
            for icx, cell_index in enumerate(block_hexcell_list):
                # set neighbour cell_index pointers
                iz = int(np.floor(icx / (nx_cells * ny_cells)))
                iy = int(np.floor((icx - (iz * nx_cells * ny_cells)) / nx_cells))
                ix = int(icx - (iz * nx_cells * ny_cells) - iy * nx_cells)

                if ix > 0:
                    left_cell_index = block_hexcell_list[icx - 1]
                    self.hexcell_list.hexcells[cell_index].neighbours["left"] = self.hexcell_list.hexcells[
                        left_cell_index
                    ]

                if ix < nx_cells - 1:
                    right_cell_index = block_hexcell_list[icx + 1]
                    self.hexcell_list.hexcells[cell_index].neighbours["right"] = self.hexcell_list.hexcells[
                        right_cell_index
                    ]

                if iy > 0:
                    front_cell_index = block_hexcell_list[icx - nx_cells]
                    self.hexcell_list.hexcells[cell_index].neighbours["front"] = self.hexcell_list.hexcells[
                        front_cell_index
                    ]

                if iy < ny_cells - 1:
                    back_cell_index = block_hexcell_list[icx + nx_cells]
                    self.hexcell_list.hexcells[cell_index].neighbours["back"] = self.hexcell_list.hexcells[
                        back_cell_index
                    ]

                if iz > 0:
                    bottom_cell_index = block_hexcell_list[icx - nx_cells * ny_cells]
                    self.hexcell_list.hexcells[cell_index].neighbours["bottom"] = self.hexcell_list.hexcells[
                        bottom_cell_index
                    ]

                if iz < nz_cells - 1:
                    top_cell_index = block_hexcell_list[icx + nx_cells * ny_cells]
                    self.hexcell_list.hexcells[cell_index].neighbours["top"] = self.hexcell_list.hexcells[
                        top_cell_index
                    ]

        print(f" Total no of cells = {_running_total_cells}")

        self.remove_duplicate_vertices()
        print(f" Total no of vertices = {len(self.hexvertex_list.hexvertices)}")

        # add in the shared neighbour search need faster method than bind neighbours
        if add_neighbours:
            self.add_neighbours_from_vertices()

        _end = time.time()
        print(f" Wall time (s) for mesh creation = {(_end-_start):.1f}")
        if _running_total_cells > 0:
            print(f" Seconds per 100k cells = {(100000*(_end-_start)/_running_total_cells):.0f}")

        # calculate quality metric - this can be slow!!
        if self.quality_metrics:
            print(" Calculating cell quality")
            if len(self.hexcell_list.hexcells) > 0:
                self.quality_list = self.quality
                max_quality = max(self.quality_list)
                min_quality = min(self.quality_list)
                avg_quality = sum(self.quality_list) / len(self.hexcell_list.hexcells)
                print(f" Cell quality = Max/Min/Av {max_quality:.1f} {min_quality:.1f} {avg_quality:.1f} ")

    def clear_mesh(self):
        """deletes the Mesh and Cell and Vertex lists"""
        self.mesh = Mesh()
        self.hexcell_list = HexCellList()
        self.hexvertex_list = HexVertexList()

    def remove_duplicate_vertices(self):
        """removes duplicates from the vertex list"""

        print(" Finding duplicate vertices")
        _nprint = 5000

        kdtree = KDTree(self.hexvertex_list.positions)
        deleted_index_list: list[int] = []

        # _start = time.time()
        # delete duplicates then add whats not deleted to new list
        for vertex in self.hexvertex_list.hexvertices:
            if vertex.index % _nprint == 0:
                print(f" vertex = {vertex.index}", end="\r")

            # if ( vertex.index not in deleted_index_list ):
            if vertex.index not in deleted_index_list:
                dup_list = kdtree.query_ball_point(vertex.position, r=constants.TOL, workers=-1)  # type: ignore
                if len(dup_list) > 1:
                    for dup_index in (dup_index for dup_index in dup_list if dup_index != vertex.index):
                        # for dup_index in dup_list:
                        # if (dup_index != vertex.index):
                        # delete the dup_index vertex
                        deleted_index_list.append(dup_index)
                        dup_vertex = self.hexvertex_list.hexvertices[dup_index]

                        for cell_index in dup_vertex.cell_indices:
                            # find cell
                            cell = self.hexcell_list.hexcells[cell_index]
                            # find vertex index in this cell
                            if dup_vertex in cell.hexvertices:
                                # TODO allow for multiple colapsed sides !!!
                                # replace vertex on cell
                                cell.hexvertices[cell.hexvertices.index(dup_vertex)] = vertex
        # _end = time.time()
        # print(f" Wall time (s) for chunker iteration = {(_end-_start):.1f}")

        print(f" No of vertices to be deleted {len(deleted_index_list)}")

        # build new vertex list if required
        if len(deleted_index_list) > 0:
            print(" Deleting vertices")
            if len(deleted_index_list) > 20000:
                print(" this will be slow - please be patient")
                est_minutes = 0.015 * len(deleted_index_list) / 60
                print(f" there will be no screen update for approx {est_minutes:.0f} minutes")

            # set deletion is 3 times faster than for loop for this operation (needs rechecked)
            # _start = time.time()
            deleted_vertices: list[Vertex] = [
                hexvertex for hexvertex in self.hexvertex_list.hexvertices if vertex.index in deleted_index_list
            ]
            s_1 = set(self.hexvertex_list.hexvertices)
            s_2 = set(deleted_vertices)
            new_vertices = s_1.difference(s_2)
            self.hexvertex_list.hexvertices = [hexvertex for hexvertex in new_vertices]
            # _end = time.time()
            # print(f" Wall time (s) for set deletion = {(_end-_start):.1f}")

            # do we have to do this ??? who cares in the indexes are not consecutive !!!
            idx = -1
            for vertex in self.hexvertex_list.hexvertices:
                idx += 1
                if idx % _nprint == 0:
                    print(f" renumbering vertices = {idx}", end="\r")
                vertex.index = idx

        return

    def add_neighbours_from_vertices(self):
        """adds neighbours to the hexcells from the cell list on each vertex"""

        print(" Adding cell neighbours")
        _nprint = 5000

        # delete duplicates then add whats not deleted to new list
        for cell in self.hexcell_list.hexcells:
            if cell.index % _nprint == 0:
                print(f" cell = {cell.index}", end="\r")

            for hexvertex in cell.hexvertices:
                for candidate_index in hexvertex.cell_indices:
                    if candidate_index != cell.index:
                        orient = [
                            key
                            for key, value in cell.neighbours.items()
                            if value == self.hexcell_list.hexcells[candidate_index]
                        ]
                        if len(orient) == 0:
                            cell.add_neighbour(self.hexcell_list.hexcells[candidate_index])

        return

    def write_vtk(self, path: str) -> None:
        """Generates a simple VTK file where each cell is a hexahedral cell"""

        print(f" Writing : {path}")
        with open(path, "w", encoding="utf-8") as output:
            header = "# vtk DataFile Version 2.0\n" + "classy_blocks hexmesh extension output\n" + "ASCII\n"

            output.write(header)

            # points
            vertices = self.hexvertex_list.hexvertices
            output.write("\nDATASET UNSTRUCTURED_GRID\n")
            output.write(f"POINTS {len(vertices)} float\n")

            for vertex in vertices:
                output.write(f"{vertex.position[0]} {vertex.position[1]} {vertex.position[2]}\n")

            # cells

            hexcells = self.hexcell_list.hexcells
            n_hexcells = len(self.hexcell_list.hexcells)
            output.write(f"\nCELLS {n_hexcells} {9*n_hexcells}\n")

            for hexcell in hexcells:
                output.write("8")
                for vertex in hexcell.hexvertices:
                    output.write(f" {vertex.index}")
                output.write("\n")

            # cell types = all type 12
            output.write(f"\nCELL_TYPES {n_hexcells}\n")
            for _ in hexcells:
                output.write("12\n")

            # cell data
            output.write(f"\nCELL_DATA {n_hexcells}\n")
            output.write("SCALARS cell_ids float 1\n")
            output.write("LOOKUP_TABLE default\n")

            for i in range(n_hexcells):
                output.write(f"{i}\n")

            # block data
            if len(self.mesh.block_list.blocks) > 1:
                output.write("SCALARS block_ids float 1\n")
                output.write("LOOKUP_TABLE default\n")

                for hexcell in hexcells:
                    output.write(f"{self.mesh.block_list.blocks.index(hexcell.block)}\n")

            if self.quality_metrics:
                output.write("SCALARS cell_quality float 1\n")
                output.write("LOOKUP_TABLE default\n")

                for i in range(n_hexcells):
                    output.write(f"{self.quality_list[i]}\n")

    @property
    def quality(self) -> list[float]:
        """Returns summed qualities of all cells in this mesh"""
        if len(self.hexcell_list.hexcells) == 0:
            return []
        return [hexcell.quality for hexcell in self.hexcell_list.hexcells]
