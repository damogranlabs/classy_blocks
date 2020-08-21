#!/usr/bin/python3
from examples.complex import helmholtz_nozzle, piping

from examples.primitive import from_points
from_points.create()

from examples.operation import loft, extrude, revolve, wedge, airfoil_2d
loft.create()
extrude.create()
revolve.create()
wedge.create()
airfoil_2d.create()

from examples.shape import elbow, cylinder
elbow.create()
cylinder.create()

from examples.complex import helmholtz_nozzle, piping
piping.create()
helmholtz_nozzle.create()