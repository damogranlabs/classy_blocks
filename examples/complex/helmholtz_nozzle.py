from classy_blocks.classes.mesh import Mesh
from classy_blocks.classes.operations import Face
from classy_blocks.classes.shapes import Cylinder, Frustum, RevolvedRing

def get_mesh():
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

    # number of cells:
    n_cells_radial = 8
    n_cells_tangential = 12
    cell_ratio = 2 # ratio between axial (flow-aligned) and radial cell size
    outer_cell_size = (2*r_nozzle)/(3*n_cells_radial)
    axial_cell_size = outer_cell_size*cell_ratio

    mesh = Mesh()

    # inlet
    x_start = 0
    x_end = l_inlet
    inlet = Cylinder([x_start, 0, 0], [x_end, 0, 0], [0, 0, r_inlet])
    inlet.set_radial_cell_count(n_cells_radial)
    inlet.set_axial_cell_count((x_end-x_start)/axial_cell_size)
    inlet.set_tangential_cell_count(n_cells_tangential)

    inlet.set_bottom_patch('inlet')
    inlet.set_outer_patch('wall')
    mesh.add(inlet)

    # nozzle
    x_start = x_end
    x_end += l_nozzle
    nozzle = Frustum([x_start, 0, 0], [x_end, 0, 0], [x_start, 0, r_inlet], r_nozzle)
    nozzle.set_axial_cell_count((x_end-x_start)/axial_cell_size)
    # the interesting part is the sharp edge: make cells denser here;
    # since we need the requested cell size at the end of the block, just use a negative size
    nozzle.grade_to_size_axial(-outer_cell_size)
    nozzle.set_outer_patch('wall')
    mesh.add(nozzle)

    # chamber: inner cylinder
    x_start = x_end
    x_end = x_end + l_chamber_inner
    chamber_inner = Cylinder([x_start, 0, 0], [x_end, 0, 0], [x_start, 0, r_nozzle])
    chamber_inner.set_axial_cell_count((x_end-x_start)/axial_cell_size)
    chamber_inner.grade_to_size_axial(outer_cell_size)
    mesh.add(chamber_inner)

    # chamber outer: ring
    ring_face = Face([
        [x_start, r_nozzle, 0],
        [x_end, r_nozzle, 0],
        [x_start + l_chamber_outer, r_chamber_outer, 0],
        [x_start, r_chamber_outer, 0]
    ])
    chamber_outer = RevolvedRing([x_start, 0, 0], [x_end, 0, 0], ring_face)
    chamber_outer.set_axial_cell_count((x_end-x_start)/axial_cell_size)
    chamber_outer.set_radial_cell_count((r_chamber_outer-r_nozzle)/axial_cell_size)

    chamber_outer.grade_to_size_radial(outer_cell_size)
    chamber_outer.set_bottom_patch('wall')
    chamber_outer.set_top_patch('wall')
    chamber_outer.set_outer_patch('wall')
    mesh.add(chamber_outer)

    # outlet pipe
    x_start = x_end
    x_end = x_end + l_outlet
    outlet = Cylinder([x_start, 0, 0], [x_end, 0, 0], [x_start, 0, r_nozzle])
    outlet.set_axial_cell_count((x_end-x_start)/axial_cell_size)
    outlet.set_outer_patch('wall')
    outlet.set_top_patch('outlet')
    mesh.add(outlet)

    # done.    
    return mesh
