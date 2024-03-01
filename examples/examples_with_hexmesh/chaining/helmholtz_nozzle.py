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
cell_size = 1.5e-3
bl_size = 0.15e-3
c2c_expansion = 1.1
axial_expansion = 2  # make cells in non-interesting places longer to save on count

mesh = cb.Mesh()

# inlet
inlet = cb.Cylinder([0, 0, 0], [l_inlet, 0, 0], [0, 0, r_inlet])
inlet.chop_axial(start_size=cell_size * axial_expansion, end_size=cell_size)
inlet.chop_tangential(start_size=cell_size)

inlet.set_start_patch("inlet")
inlet.set_outer_patch("wall")
mesh.add(inlet)

# nozzle
nozzle = cb.Frustum.chain(inlet, l_nozzle, r_nozzle)
# cell sizing: make sure bl_size is correct here
nozzle.chop_axial(length_ratio=0.5, start_size=cell_size * axial_expansion, end_size=cell_size)
nozzle.chop_axial(length_ratio=0.5, start_size=cell_size, end_size=bl_size)
nozzle.chop_radial(end_size=bl_size, c2c_expansion=1 / c2c_expansion)
nozzle.set_outer_patch("wall")
mesh.add(nozzle)

# chamber: inner cylinder
chamber_inner = cb.Cylinder.chain(nozzle, l_chamber_inner)
# create smaller cells at inlet and outlet but leave them bigger in the middle;
chamber_inner.chop_axial(length_ratio=0.25, start_size=bl_size, end_size=cell_size)
chamber_inner.chop_axial(length_ratio=0.25, start_size=cell_size, end_size=cell_size * axial_expansion)

chamber_inner.chop_axial(length_ratio=0.25, start_size=cell_size * axial_expansion, end_size=cell_size)
chamber_inner.chop_axial(length_ratio=0.25, start_size=cell_size, end_size=bl_size)
mesh.add(chamber_inner)

# chamber outer: expanded ring; the end face will be moved when the mesh is assembled
chamber_outer = cb.ExtrudedRing.expand(chamber_inner, r_chamber_outer - r_inlet)
chamber_outer.chop_radial(length_ratio=0.5, start_size=bl_size, c2c_expansion=c2c_expansion)
chamber_outer.chop_radial(length_ratio=0.5, end_size=bl_size, c2c_expansion=1 / c2c_expansion)
chamber_outer.set_start_patch("wall")
chamber_outer.set_end_patch("wall")
chamber_outer.set_outer_patch("wall")
mesh.add(chamber_outer)

# translate outer points of outer chamber (and edges) to get
# that inverted cone at the end;
# this could also be done by a RevolvedRing
for face in chamber_outer.sketch_2.faces:
    for i in (1, 2):
        face.points[i].translate([l_chamber_outer - l_chamber_inner, 0, 0])

    face.add_edge(1, cb.Origin([l_inlet + l_nozzle + l_chamber_outer, 0, 0]))

# outlet pipe
outlet = cb.Cylinder.chain(chamber_inner, l_outlet)
outlet.chop_axial(length_ratio=0.5, start_size=bl_size, end_size=cell_size)
outlet.chop_axial(length_ratio=0.5, start_size=cell_size, end_size=cell_size * axial_expansion)
outlet.set_outer_patch("wall")
outlet.set_end_patch("outlet")
mesh.add(outlet)

# mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")

hexmesh = cb.HexMesh(mesh)
hexmesh.write_vtk(os.path.join("helmholtz_nozzle.vtk"))
