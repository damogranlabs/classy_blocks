import os

import classy_blocks as cb

geometry = {
    "terrain": [
        "type triSurfaceMesh",
        "name terrain",
        'file "terrain.stl"',
    ],
    "left_wall": [
        "type       searchablePlane",
        "planeType  pointAndNormal",
        "point      (-1 0  0)",
        "normal     (1  0  0)",
        "pointAndNormalDict { point (-1 0 0); normal (1 0 0); }",  # ESI version
    ],
    "front_wall": [
        "type       searchablePlane",
        "planeType  pointAndNormal",
        "point      (0 -1  0)",
        "normal     (0  1  0)",
        "pointAndNormalDict { point (0 -1 0); normal (0 1 0); }",  # ESI version
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
# box = cb.Box([-0.9, -0.9, -0.9], [1.0, 1.0, 1.0])
# box = cb.Box([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])
box = cb.Box([-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])

# project misplaced vertices

box.project_corner(0, "sphere")
box.project_corner(1, "sphere")
box.project_corner(2, "sphere")
box.project_corner(3, "sphere")

box.project_corner(4, "sphere")
box.project_corner(5, "sphere")
box.project_corner(6, "sphere")
box.project_corner(7, "sphere")

# project a edge to geometry;

box.project_edge(0, 1, "sphere")
box.project_edge(1, 2, "sphere")
box.project_edge(3, 2, "sphere")
box.project_edge(0, 3, "sphere")

box.project_edge(4, 5, "sphere")
box.project_edge(5, 6, "sphere")
box.project_edge(7, 6, "sphere")
box.project_edge(4, 7, "sphere")

box.project_edge(0, 4, "sphere")
box.project_edge(1, 5, "sphere")
box.project_edge(2, 6, "sphere")
box.project_edge(3, 7, "sphere")

box.project_side("right", "sphere")

box.project_side("bottom", "sphere")

box.project_side("top", "sphere")

box.project_side("left", "sphere")

box.project_side("right", "sphere")

box.project_side("front", "sphere")

box.project_side("back", "sphere")

for axis in (0, 1, 2):
    box.chop(axis, count=20)

box.set_patch("bottom", "terrain")

mesh.add(box)

mesh.set_default_patch("atmosphere", "patch")
mesh.add_geometry(geometry)

# mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")

hexmesh = cb.HexMesh(mesh, quality_metrics=True)
hexmesh.write_vtk(os.path.join("project_sphere_mesh.vtk"))
