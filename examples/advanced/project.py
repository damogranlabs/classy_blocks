import os

import classy_blocks as cb

terrain_surface = cb.SearchableTriSurface("terrain", "terrain.stl")
left_wall = cb.SearchablePlanePointAndNormal("left_wall", [-1, 0, 0], [1, 0, 0])
front_wall = cb.SearchablePlanePointAndNormal("front_wall", [0, -1, 0], [0, 1, 0])

mesh = cb.Mesh()

# 'miss' the vertices deliberately;
# project them to geometry later
box = cb.Box([-0.9, -0.9, -0.9], [1.0, 1.0, 1.0])

# project misplaced vertices
box.project_corner(0, ["terrain", "left_wall", "front_wall"])
box.project_corner(1, "terrain")
box.project_corner(2, "terrain")

# project a face to geometry;
# when using Loft/Extrude/Revolve, you could specify
# those when creating a Face; you'd still have to
# project other sides this way
box.project_side("bottom", "terrain")

# projection of an edge to a surface will move it in various directions,
# depending on the geometry, distorting the box's side
box.project_edge(0, 1, "terrain")

# to avoid that, project an edge to two surfaces;
# this will make it stick to their intersection
box.project_edge(3, 0, ["terrain", "left_wall"])

# an edge can remain 'not projected' but this can cause
# bad quality cells if geometries differ enough
# extrude.block.project_edge(1, 2, 'terrain')
# extrude.block.project_edge(2, 3, 'terrain')
# extrude.block.project_edge(3, 0, 'terrain')

for axis in (0, 1, 2):
    box.chop(axis, count=20)

box.set_patch("bottom", "terrain")

mesh.add(box)

mesh.set_default_patch("atmosphere", "patch")
mesh.add_geometry(terrain_surface)
mesh.add_geometry(left_wall)
mesh.add_geometry(front_wall)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
