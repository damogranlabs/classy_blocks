import os
import classy_blocks as cb

# A nozzle with a chamber that produces self-induced oscillations.
# See helmholtz_nozzle.svg for geometry explanation.

# Note that this example could be greatly simplified with the new
# .chain()/.expand()/.contract() methods; it was written before.

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

# number of cells:
n_cells_radial = 8
n_cells_tangential = 12
cell_ratio = 4  # ratio between axial (flow-aligned) and radial cell size
outer_cell_size = (2 * r_nozzle) / (3 * n_cells_radial)
axial_cell_size = outer_cell_size * cell_ratio
c2c_expansion = 1.2

mesh = cb.Mesh()

# inlet
inlet = cb.Cylinder([0, 0, 0], [l_inlet, 0, 0], [0, 0, r_inlet])
inlet.chop_radial(count=n_cells_radial)
inlet.chop_axial(start_size=axial_cell_size)
inlet.chop_tangential(count=n_cells_tangential)

inlet.set_start_patch("inlet")
inlet.set_outer_patch("wall")
mesh.add(inlet)

# nozzle
nozzle = cb.Frustum.chain(inlet, l_nozzle, r_nozzle)
# the interesting part is the sharp edge: make cells denser here;
# since we need the requested cell size at the end of the block, just use a negative size
nozzle.chop_axial(start_size=axial_cell_size, end_size=outer_cell_size)
nozzle.set_outer_patch("wall")
mesh.add(nozzle)

# chamber: inner cylinder
chamber_inner = cb.Cylinder.chain(nozzle, l_chamber_inner)
# create smaller cells at inlet and outlet but leave them bigger in the middle;
chamber_inner.chop_axial(length_ratio=0.5, start_size=outer_cell_size, end_size=axial_cell_size)
chamber_inner.chop_axial(length_ratio=0.5, start_size=axial_cell_size, end_size=outer_cell_size)
mesh.add(chamber_inner)

# chamber outer: ring
x_start = l_inlet + l_nozzle
x_end = x_start + l_chamber_inner
chamber_outer = cb.RevolvedRing(
    [x_start, 0, 0],
    [x_end, 0, 0],
    cb.Face(
        [  # ring face
            [x_start, r_nozzle, 0],
            [x_end, r_nozzle, 0],
            [x_start + l_chamber_outer, r_chamber_outer, 0],
            [x_start, r_chamber_outer, 0],
        ]
    ),
)
chamber_outer.chop_radial(length_ratio=0.5, start_size=outer_cell_size, c2c_expansion=c2c_expansion)
chamber_outer.chop_radial(length_ratio=0.5, end_size=outer_cell_size, c2c_expansion=1 / c2c_expansion)
chamber_outer.set_start_patch("wall")
chamber_outer.set_end_patch("wall")
chamber_outer.set_outer_patch("wall")
mesh.add(chamber_outer)

# outlet pipe
outlet = cb.Cylinder.chain(chamber_inner, l_outlet)
outlet.chop_axial(start_size=outer_cell_size, end_size=axial_cell_size)
outlet.set_outer_patch("wall")
outlet.set_end_patch("outlet")
mesh.add(outlet)

mesh.write(os.path.join("..", "case", "system", "blockMeshDict"), debug_path="debug.vtk")
