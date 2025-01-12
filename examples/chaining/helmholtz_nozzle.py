import os

import classy_blocks as cb

# A nozzle with a chamber that produces self-induced oscillations.
# See helmholtz_nozzle.svg for geometry explanation.

# geometry data (all dimensions in meters):
# inlet pipe
r_inlet = 10e-3
l_inlet = 50e-3

# nozzle
r_nozzle = 6e-3
l_nozzle = 20e-3

# chamber
l_chamber_inner = 100e-3
l_chamber_outer = 105e-3
r_chamber_outer = 20e-3

# outlet
l_outlet = 80e-3

# cell sizing
first_cell_size = 0.01e-3
bulk_cell_size = 1.5e-3

mesh = cb.Mesh()

# inlet
inlet = cb.Cylinder([0, 0, 0], [l_inlet, 0, 0], [0, 0, r_inlet])

inlet.set_start_patch("inlet")
inlet.set_outer_patch("wall")
mesh.add(inlet)

# nozzle
nozzle = cb.Frustum.chain(inlet, l_nozzle, r_nozzle)
nozzle.set_outer_patch("wall")
mesh.add(nozzle)

# chamber: inner cylinder
chamber_inner = cb.Cylinder.chain(nozzle, l_chamber_inner)
mesh.add(chamber_inner)

# chamber outer: expanded ring; the end face will be moved when the mesh is assembled
chamber_outer = cb.ExtrudedRing.expand(chamber_inner, r_chamber_outer - r_inlet)

# translate outer points of outer chamber (and edges) to get
# that inverted cone at the end;
# this could also be done by a RevolvedRing
for face in chamber_outer.sketch_2.faces:
    for i in (1, 2):
        face.points[i].translate([l_chamber_outer - l_chamber_inner, 0, 0])

    face.add_edge(1, cb.Origin([l_inlet + l_nozzle + l_chamber_outer, 0, 0]))

chamber_outer.set_start_patch("wall")
chamber_outer.set_end_patch("wall")
chamber_outer.set_outer_patch("wall")
mesh.add(chamber_outer)

# outlet pipe
outlet = cb.Cylinder.chain(chamber_inner, l_outlet)
outlet.set_outer_patch("wall")
outlet.set_end_patch("outlet")
mesh.add(outlet)

mesh.modify_patch("wall", "wall")

grader = cb.InflationGrader(mesh, first_cell_size, bulk_cell_size)
grader.grade()

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
