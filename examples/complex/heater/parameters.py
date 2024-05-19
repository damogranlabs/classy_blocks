from classy_blocks.util import functions as f

# geometry
heater_diameter = 10
heater_length = 50
bend_radius = 3 * heater_diameter
domain_size = 50

# cell sizing
solid_cell_size = 1
fluid_cell_size = 0.5
first_cell_size = 0.05
c2c_expansion = 1.2

# mesh parameters (do not edit)
wrapping_height = (bend_radius + heater_diameter / 2) / 2

# coordinates, decomposed into grid, a.k.a. 'levels';
# see heater.svg for explanation
xlv = [
    -heater_length,
    0,
    bend_radius - wrapping_height / 2,
    bend_radius - heater_diameter / 2,
    bend_radius,
    bend_radius + heater_diameter / 2,
    bend_radius + wrapping_height / 2,
    domain_size,
]

ylv = [
    -domain_size,
    -bend_radius - wrapping_height / 2,
    -bend_radius - heater_diameter / 2,
    -bend_radius,
    -bend_radius + wrapping_height / 2,
]
zlv = [-wrapping_height / 2, wrapping_height / 2]

heater_start_point = f.vector(xlv[0], ylv[3], 0)
wrapping_corner_point = heater_start_point + f.vector(0, wrapping_height / 2, -wrapping_height / 2)
