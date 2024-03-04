"""functions specifically used by read and interogate stl files specified in geometry"""

from pathlib import Path, PurePath

from scipy.spatial import KDTree
from stl import Mesh as stlMesh

from classy_blocks import Mesh
from classy_blocks.types import NPPointType


def check_file_exists(file_object):
    # return logical true if file exists
    _exists = False

    if isinstance(file_object, Path):
        # Path object can use is_file() function
        _exists = file_object.is_file()
    elif isinstance(file_object, PurePath):
        # PurePath object can use exists() funtion
        _exists = Path(file_object).is_file()
    elif isinstance(file_object, str):
        # string
        _exists = Path(str(file_object)).is_file()

    return _exists


def read_stl(stl_file: str) -> tuple[stlMesh | None, KDTree | None]:
    """reads an stl file"""

    if len(stl_file) == 0:
        return (None, None)

    _stl_file_full: PurePath | str = stl_file
    if not check_file_exists(_stl_file_full):
        # file not found look in current working directory
        _stl_file_full = PurePath(Path.cwd(), stl_file)
        if not check_file_exists(_stl_file_full):
            # file still not found look in the blockmesh folder as subfolder
            _stl_file_full = PurePath(Path.cwd().parent, "case", "constant", stl_file)
            if not check_file_exists(_stl_file_full):
                # file still not found look in the blockmesh folder as subfolder
                _stl_file_full = PurePath(Path.cwd().parent, "case", "constant", "geometry", stl_file)
                if not check_file_exists(_stl_file_full):
                    # file still not found look in another blockmesh folder as subfolder of parent
                    _stl_file_full = PurePath(Path.cwd().parent, "case", "constant", "triSurface", stl_file)
                    if not check_file_exists(_stl_file_full):
                        # out of options
                        print(f" stl file {stl_file} not found")
                        return (None, None)
    # file found - read it
    # Using an existing stl file:
    stl_msh = stlMesh.from_file(_stl_file_full)
    # Get the number of faces
    num_faces = stl_msh.vectors.shape[0]
    # load all the points into a kdtree
    stl_points: list[NPPointType] = []
    for patch in stl_msh:
        stl_points.append(patch[0:3])
        stl_points.append(patch[3:6])
        stl_points.append(patch[6:9])
    stl_tree = KDTree(stl_points)

    # num_vertices = len(stl_msh.vectors)
    print(f" Read file : {stl_file} complete {num_faces} faces ")

    return (stl_msh, stl_tree)  # type: ignore


def read_geometry(mesh: Mesh) -> list[dict[str, stlMesh] | dict[str, KDTree]]:
    """returns disctionary of stl mesh objects in stl specified in mesh geometry"""

    stlmesh_meshs: dict[str, stlMesh] = {}
    stlmesh_trees: dict[str, KDTree] = {}

    if len(mesh.geometry_list.geometry) > 0:
        # check if the geometry disctionary needs us to read a file
        for geometry_name in mesh.geometry_list.geometry:
            geom_vars = mesh.geometry_list.geometry[geometry_name]
            trisurfacemesh = False
            # find type
            for geom_var in geom_vars:
                if geom_var.lower().find("type") != -1:
                    if geom_var.lower().find("trisurfacemesh") != -1:
                        trisurfacemesh = True
            # find file
            if trisurfacemesh:
                for geom_var in geom_vars:
                    _start = geom_var.lower().find("file")
                    if _start != -1:
                        _end = len(geom_var)
                        stl_file_name = geom_var[_start + 4 : _end].replace("'", " ").replace('"', " ").strip()

                        (stl_msh, stl_tree) = read_stl(stl_file_name)
                        if stl_msh is not None:
                            stlmesh_meshs[geometry_name] = stl_msh
                        if stl_tree is not None:
                            stlmesh_trees[geometry_name] = stl_tree

    return [stlmesh_meshs, stlmesh_trees]
