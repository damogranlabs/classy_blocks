import os

import classy_blocks as cb

geometry = {
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
    "top_wall1": [
        "type       searchablePlane",
        "planeType  pointAndNormal",
        "point      (0.5 0.5  0.5)",
        "normal     (0  0  1)",
    ],
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
    "top_wall2": [
        "type       searchablePlane",
        "planeType  embeddedPoints",
        "point1     (-1 -1  0.0  )",
        "point2     ( 1 -1  0.2 )",
        "point3     (-1  1  1.0 )",
    ],
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
    "top_wall3": [
        "type       searchablePlane",
        "planeType  planeEquation",
        " a    1",
        " b    1",
        " c    1",
        " d    1.",
    ],
    "sphere": [
        "type       searchableSphere",
        "centre          (0 0 0)",
        "radius          1.0",
    ],
}

mesh = cb.Mesh()

# 'miss' the vertices deliberately;
# project them to geometry later
box = cb.Box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])

# project misplaced vertices
box.project_corner(4, ["top_wall3"])
box.project_corner(5, ["top_wall3"])
box.project_corner(6, ["top_wall3"])
box.project_corner(7, ["top_wall3"])

# project a face to geometry;
# when using Loft/Extrude/Revolve, you could specify
# those when creating a Face; you'd still have to
# project other sides this way
box.project_side("top", ["top_wall3"])

# projection of an edge to a surface will move it in various directions,
# depending on the geometry, distorting the box's side
box.project_edge(4, 5, ["top_wall3"])
box.project_edge(5, 6, ["top_wall3"])
box.project_edge(6, 7, ["top_wall3"])
box.project_edge(7, 4, ["top_wall3"])

for axis in (0, 1, 2):
    box.chop(axis, count=20)

mesh.add(box)

mesh.set_default_patch("atmosphere", "patch")
mesh.add_geometry(geometry)

# mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")

hexmesh = cb.HexMesh(mesh)
hexmesh.write_vtk(os.path.join("project_plane.vtk"))
