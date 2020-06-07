#!/usr/bin/python3
from examples.complex import helmholtz_nozzle, piping

from examples.primitive import from_points
from_points.create()

from examples.operation import loft, extrude, revolve, wedge
loft.create()
extrude.create()
revolve.create()
wedge.create()

from examples.shape import elbow, cylinder
elbow.create()
cylinder.create()

from examples.complex import helmholtz_nozzle, piping
helmholtz_nozzle.create()
piping.create()